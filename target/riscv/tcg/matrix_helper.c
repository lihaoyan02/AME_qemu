/*
 * Attached Matrix Extension helpers.
 */

#include "qemu/osdep.h"

#include <math.h>
#include "cpu.h"
#include "accel/tcg/cpu-ldst.h"
#include "accel/tcg/probe.h"
#include "exec/helper-proto.h"
#include "exec/target_page.h"
#include "internals.h"
static inline void ame_require_ms(CPURISCVState *env, uintptr_t ra)
{
#ifndef CONFIG_USER_ONLY
    if ((env->mstatus & MSTATUS_MS) == 0) {
        riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, ra);
    }
#endif
}

static inline void ame_require_enabled(CPURISCVState *env, uintptr_t ra)
{
    ame_require_ms(env, ra);
    if (!(env->ameown & AMEOWN_OWNED)) {
        riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, ra);
    }
}

void HELPER(ame_check_enabled)(CPURISCVState *env)
{
    ame_require_enabled(env, GETPC());
}

#include "fpu/softfloat.h"

enum {
    AME_MMULACC = 0x4a,
    AME_MMULACCNEG = 0x4b,
    AME_MMULATACC = 0x4d,
    AME_MMULATACCNEG = 0x4e,
    AME_MMULBTACC = 0x4f,
    AME_MMULBTACCNEG = 0x50,
    AME_MAND = 0x1a,
    AME_MANDNOT = 0x1c,
    AME_MOR = 0x1e,
    AME_MORNOT = 0x20,
    AME_MSLL = 0x22,
    AME_MSRA = 0x24,
    AME_MSRL = 0x26,
    AME_MXOR = 0x28,
    AME_MCMOVGE = 0x2a,
    AME_MCMOVLT = 0x2b,
    AME_MCMPGE = 0x2c,
    AME_MCMPGE_X = 0x2d,
    AME_MCMPLT = 0x2e,
    AME_MCMPLT_X = 0x2f,
    AME_MSELGE = 0x30,
    AME_MSELLT = 0x31,
    AME_MCOLBCAST = 0x32,
    AME_MCOLGATHER = 0x33,
    AME_MCOLSHIFT = 0x34,
    AME_MROWBCAST = 0x35,
    AME_MROWGATHER = 0x36,
    AME_MROWSHIFT = 0x37,
    AME_MROWZIP = 0x38,
    AME_MROWSCATADD = 0x39,
    AME_MCOLSCATADD = 0x3a,
    AME_MROWSCATMAX = 0x3b,
    AME_MCOLSCATMAX = 0x3c,
    AME_MCOLUNZIP = 0x61,
    AME_MCOLZIP = 0x62,
    AME_MROWUNZIP = 0x63,
    AME_MABS = 0xa24,
    AME_MABSDIFF = 0x00,
    AME_MADD = 0x02,
    AME_MHDIFF = 0x04,
    AME_MMAX = 0x06,
    AME_MMEAN = 0x08,
    AME_MMIN = 0x0a,
    AME_MMUL = 0x0c,
    AME_MMULACC_EW = 0x0e,
    AME_MMULACCNEG_EW = 0x10,
    AME_MMULADD = 0x12,
    AME_MMULNEG = 0x14,
    AME_MMULSUB = 0x16,
    AME_MSUB = 0x18,
};

static bool ame_integer_datatype_valid(uint32_t datatype)
{
    unsigned size = datatype & 0xff;

    return !(datatype & ((1u << 31) | (1u << 8))) &&
           !(datatype & AME_DATATYPE_RESERVED_MASK) &&
           (size == 4 || size == 8 || size == 16 || size == 32 ||
            size == 64 || size == 128);
}

static bool ame_float_datatype_valid(uint32_t datatype)
{
    unsigned exp_width = (datatype >> 26) & 0x1f;
    unsigned size = datatype & 0xff;

    if ((datatype & (1u << 31)) || !(datatype & (1u << 8)) ||
        (datatype & 0x000ffe00) || ((datatype >> 22) & 0xf) > 5) {
        return false;
    }
    return (size == 16 && (exp_width == 5 || exp_width == 8)) ||
           (size == 32 && exp_width == 8) ||
           (size == 64 && exp_width == 11);
}

static bool ame_datatype_valid(uint32_t datatype)
{
    return ame_integer_datatype_valid(datatype) ||
           ame_float_datatype_valid(datatype);
}

/* Numerical conversions currently implement IEEE Inf/Denorm=1 only. */
static bool ame_conversion_type_supported(uint32_t datatype)
{
    return ame_integer_datatype_valid(datatype) ||
           (ame_float_datatype_valid(datatype) &&
            (datatype & (BIT(21) | BIT(20))) == (BIT(21) | BIT(20)));
}

static void ame_float_rounding_from_dtype(float_status *fs,
                                          uint32_t datatype);

static unsigned ame_pack_factor(CPURISCVState *env, unsigned size)
{
    return size < riscv_ame_unit_bits(env) ?
           riscv_ame_unit_bits(env) / size : 1;
}

static unsigned ame_group_size(CPURISCVState *env, unsigned size)
{
    return MAX(1, DIV_ROUND_UP(size, riscv_ame_unit_bits(env)));
}

static __uint128_t ame_load_bits(const uint8_t *p, unsigned bit_offset,
                                 unsigned bits)
{
    __uint128_t value = 0;

    for (unsigned i = 0; i < bits; i++) {
        value |= (__uint128_t)((p[(bit_offset + i) / 8] >>
                                ((bit_offset + i) % 8)) & 1) << i;
    }
    return value;
}

static void ame_store_bits(uint8_t *p, unsigned bit_offset, unsigned bits,
                           __uint128_t value)
{
    for (unsigned i = 0; i < bits; i++) {
        unsigned byte = (bit_offset + i) / 8;
        unsigned shift = (bit_offset + i) % 8;

        p[byte] = (p[byte] & ~(1u << shift)) |
                  (((value >> i) & 1) << shift);
    }
}

enum {
    AME_MEM_1R,
    AME_MEM_CM,
    AME_MEM_RM,
    AME_MEM_ST,
    AME_MEM_TST,
};

typedef struct AMEMemoryOperand {
    unsigned size;
    unsigned pack_factor;
    unsigned group_size;
    unsigned bytes;
} AMEMemoryOperand;

static bool ame_memory_operand(CPURISCVState *env, unsigned reg,
                               unsigned layout, AMEMemoryOperand *op,
                               uintptr_t ra)
{
    uint32_t datatype;

    if (reg >= riscv_ame_mregs(env) || layout > AME_MEM_TST) {
        riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, ra);
    }
    if (layout == AME_MEM_1R) {
        op->size = riscv_ame_unit_bits(env);
        op->pack_factor = 1;
        op->group_size = 1;
        op->bytes = riscv_ame_mreg_bytes(env);
        return true;
    }

    datatype = env->md[reg];
    if (!ame_datatype_valid(datatype)) {
        env->amestatus |= AMESTATUS_UN;
        return false;
    }
    op->size = datatype & 0xff;
    op->pack_factor = ame_pack_factor(env, op->size);
    op->group_size = ame_group_size(env, op->size);
    op->bytes = DIV_ROUND_UP(op->pack_factor * riscv_ame_nelem(env) *
                             op->size, 8);

    if (reg % op->group_size ||
        reg + op->group_size > riscv_ame_mregs(env)) {
        riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, ra);
    }
    return true;
}

static void ame_check_memory_alignment(CPURISCVState *env, target_ulong base,
                                       unsigned size, bool store,
                                       uintptr_t ra)
{
    unsigned element_bytes = DIV_ROUND_UP(size, 8);

    if (base % element_bytes) {
        env->badaddr = base;
        riscv_raise_exception(env, store ? RISCV_EXCP_STORE_AMO_ADDR_MIS :
                                           RISCV_EXCP_LOAD_ADDR_MIS, ra);
    }
}

static void ame_transpose_square_layout(CPURISCVState *env, uint8_t *dest,
                                        const uint8_t *source,
                                        const AMEMemoryOperand *op)
{
    unsigned n = riscv_ame_n(env);
    memset(dest, 0, op->bytes);
    for (unsigned square = 0; square < op->pack_factor; square++) {
        for (unsigned row = 0; row < n; row++) {
            for (unsigned column = 0; column < n; column++) {
                unsigned row_index = square * riscv_ame_nelem(env) +
                                     row * n + column;
                unsigned column_index = square * riscv_ame_nelem(env) +
                                        column * n + row;
                __uint128_t value = ame_load_bits(
                    source, row_index * op->size, op->size);

                ame_store_bits(dest, column_index * op->size,
                               op->size, value);
            }
        }
    }
}

static unsigned ame_stride_segment_count(CPURISCVState *env,
                                         const AMEMemoryOperand *op,
                                         bool transpose)
{
    return transpose ? op->pack_factor * riscv_ame_n(env) : riscv_ame_n(env);
}

static unsigned ame_stride_segment_bytes(CPURISCVState *env,
                                         const AMEMemoryOperand *op,
                                         bool transpose)
{
    unsigned elements = riscv_ame_n(env) * (transpose ? 1 : op->pack_factor);

    return DIV_ROUND_UP(elements * op->size, 8);
}

/*
 * Convert between the architectural row-major M bit string and the sequence
 * of separately byte-rounded strided segments from Ztt section 2.6.
 */
static void ame_stride_layout(CPURISCVState *env, uint8_t *segments,
                              uint8_t *registers,
                              const AMEMemoryOperand *op, bool transpose,
                              bool load)
{
    unsigned n = riscv_ame_n(env);
    unsigned segment_bytes = ame_stride_segment_bytes(env, op, transpose);

    if (!load) {
        memset(segments, 0,
               ame_stride_segment_count(env, op, transpose) * segment_bytes);
    } else {
        memset(registers, 0, op->bytes);
    }

    for (unsigned square = 0; square < op->pack_factor; square++) {
        for (unsigned row = 0; row < n; row++) {
            for (unsigned column = 0; column < n; column++) {
                unsigned reg_bit = (square * riscv_ame_nelem(env) + row * n +
                                    column) * op->size;
                unsigned segment = transpose ? square * n + column : row;
                unsigned segment_bit = transpose ? row * op->size :
                    (square * n + column) * op->size;
                uint8_t *segment_base = segments + segment * segment_bytes;
                __uint128_t value;

                if (load) {
                    value = ame_load_bits(segment_base, segment_bit, op->size);
                    ame_store_bits(registers, reg_bit, op->size, value);
                } else {
                    value = ame_load_bits(registers, reg_bit, op->size);
                    ame_store_bits(segment_base, segment_bit, op->size, value);
                }
            }
        }
    }
}

typedef struct AMEStoreChunk {
    void *host;
    unsigned offset;
    unsigned bytes;
} AMEStoreChunk;

static void ame_store_range_atomic(CPURISCVState *env, target_ulong base,
                                   const uint8_t *data, unsigned bytes,
                                   int mmu_idx, uintptr_t ra)
{
    AMEStoreChunk chunks[AME_MAX_ACC_REG_BYTES];
    unsigned nchunks = 0;
    unsigned offset = 0;

    while (offset < bytes) {
        target_ulong address = base + offset;
        vaddr adjusted = adjust_addr(env, address);
        unsigned page_bytes = -(adjusted | TARGET_PAGE_MASK);
        unsigned chunk = MIN(bytes - offset, page_bytes);
        void *host = probe_write(env, adjusted, chunk, mmu_idx, ra);

        if (!host) {
            /* Generic MMIO writes cannot be rolled back. */
            env->badaddr = adjusted;
            riscv_raise_exception(env, RISCV_EXCP_STORE_AMO_ACCESS_FAULT, ra);
        }
        chunks[nchunks++] = (AMEStoreChunk) {
            .host = host,
            .offset = offset,
            .bytes = chunk,
        };
        offset += chunk;
    }

    for (unsigned i = 0; i < nchunks; i++) {
        memcpy(chunks[i].host, data + chunks[i].offset, chunks[i].bytes);
    }
}

static void ame_store_segments_atomic(CPURISCVState *env, target_ulong base,
                                      target_ulong stride, const uint8_t *data,
                                      unsigned segments, unsigned segment_bytes,
                                      int mmu_idx, uintptr_t ra)
{
    AMEStoreChunk chunks[AME_MAX_ACC_REG_BYTES];
    unsigned nchunks = 0;

    /* Probe every segment before exposing any explicit data-memory write. */
    for (unsigned segment = 0; segment < segments; segment++) {
        target_ulong segment_base = base + (target_ulong)segment * stride;
        unsigned offset = 0;

        while (offset < segment_bytes) {
            target_ulong address = segment_base + offset;
            vaddr adjusted = adjust_addr(env, address);
            unsigned page_bytes = -(adjusted | TARGET_PAGE_MASK);
            unsigned chunk = MIN(segment_bytes - offset, page_bytes);
            void *host = probe_write(env, adjusted, chunk, mmu_idx, ra);

            if (!host) {
                env->badaddr = adjusted;
                riscv_raise_exception(env, RISCV_EXCP_STORE_AMO_ACCESS_FAULT,
                                      ra);
            }
            g_assert(nchunks < ARRAY_SIZE(chunks));
            chunks[nchunks++] = (AMEStoreChunk) {
                .host = host,
                .offset = segment * segment_bytes + offset,
                .bytes = chunk,
            };
            offset += chunk;
        }
    }

    /* Later segments win when guest address ranges overlap. */
    for (unsigned i = 0; i < nchunks; i++) {
        memcpy(chunks[i].host, data + chunks[i].offset, chunks[i].bytes);
    }
}

void HELPER(ame_mls)(CPURISCVState *env, uint32_t md, target_ulong base,
                     uint32_t layout)
{
    ame_require_enabled(env, GETPC());
    uint8_t memory[AME_MAX_M_FILE_BYTES];
    uint8_t registers[AME_MAX_M_FILE_BYTES];
    AMEMemoryOperand op;
    uintptr_t ra = GETPC();

    if (!ame_memory_operand(env, md, layout, &op, ra)) {
        return;
    }
    ame_check_memory_alignment(env, base, op.size, false, ra);

    for (unsigned i = 0; i < op.bytes; i++) {
        memory[i] = cpu_ldub_data_ra(env, adjust_addr(env, base + i), ra);
    }
    if (layout == AME_MEM_CM) {
        ame_transpose_square_layout(env, registers, memory, &op);
    } else {
        memcpy(registers, memory, op.bytes);
    }
    memcpy(riscv_ame_mreg(env, md), registers, op.bytes);
}

void HELPER(ame_mss)(CPURISCVState *env, uint32_t ms1, target_ulong base,
                     uint32_t layout)
{
    ame_require_enabled(env, GETPC());
    uint8_t memory[AME_MAX_M_FILE_BYTES];
    uint8_t registers[AME_MAX_M_FILE_BYTES];
    AMEMemoryOperand op;
    int mmu_idx = riscv_env_mmu_index(env, false);
    uintptr_t ra = GETPC();

    if (!ame_memory_operand(env, ms1, layout, &op, ra)) {
        return;
    }
    ame_check_memory_alignment(env, base, op.size, true, ra);

    memcpy(registers, riscv_ame_mreg(env, ms1), op.bytes);
    if (layout == AME_MEM_CM) {
        ame_transpose_square_layout(env, memory, registers, &op);
    } else {
        memcpy(memory, registers, op.bytes);
    }

    /* Validate and collect the complete transfer before making it visible. */
    ame_store_range_atomic(env, base, memory, op.bytes, mmu_idx, ra);
}

void HELPER(ame_mls_stride)(CPURISCVState *env, uint32_t md,
                            target_ulong base, target_ulong stride,
                            uint32_t layout)
{
    ame_require_enabled(env, GETPC());
    uint8_t segments[AME_MAX_ACC_REG_BYTES];
    uint8_t registers[AME_MAX_ACC_REG_BYTES];
    AMEMemoryOperand op;
    uintptr_t ra = GETPC();
    bool transpose = layout == AME_MEM_TST;
    unsigned count, bytes;

    if (layout != AME_MEM_ST && layout != AME_MEM_TST) {
        riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, ra);
    }
    if (!ame_memory_operand(env, md, layout, &op, ra)) {
        return;
    }
    ame_check_memory_alignment(env, base, op.size, false, ra);
    count = ame_stride_segment_count(env, &op, transpose);
    bytes = ame_stride_segment_bytes(env, &op, transpose);

    for (unsigned segment = 0; segment < count; segment++) {
        target_ulong segment_base = base + (target_ulong)segment * stride;

        for (unsigned byte = 0; byte < bytes; byte++) {
            segments[segment * bytes + byte] = cpu_ldub_data_ra(
                env, adjust_addr(env, segment_base + byte), ra);
        }
    }
    ame_stride_layout(env, segments, registers, &op, transpose, true);
    memcpy(riscv_ame_mreg(env, md), registers, op.bytes);
}

void HELPER(ame_mss_stride)(CPURISCVState *env, uint32_t ms1,
                            target_ulong base, target_ulong stride,
                            uint32_t layout)
{
    ame_require_enabled(env, GETPC());
    uint8_t segments[AME_MAX_ACC_REG_BYTES];
    uint8_t registers[AME_MAX_ACC_REG_BYTES];
    AMEMemoryOperand op;
    int mmu_idx = riscv_env_mmu_index(env, false);
    uintptr_t ra = GETPC();
    bool transpose = layout == AME_MEM_TST;
    unsigned count, bytes;

    if (layout != AME_MEM_ST && layout != AME_MEM_TST) {
        riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, ra);
    }
    if (!ame_memory_operand(env, ms1, layout, &op, ra)) {
        return;
    }
    ame_check_memory_alignment(env, base, op.size, true, ra);
    count = ame_stride_segment_count(env, &op, transpose);
    bytes = ame_stride_segment_bytes(env, &op, transpose);
    memcpy(registers, riscv_ame_mreg(env, ms1), op.bytes);
    ame_stride_layout(env, segments, registers, &op, transpose, false);
    ame_store_segments_atomic(env, base, stride, segments, count, bytes,
                              mmu_idx, ra);
}

void HELPER(ame_asettyp)(void *acc, uint32_t datatype, CPURISCVState *env)
{
    ame_require_enabled(env, GETPC());
    unsigned reg = ((uint8_t *)acc - env->acc) / riscv_ame_acc_bytes(env);

    if (!ame_datatype_valid(datatype)) {
        env->amestatus |= AMESTATUS_UN;
        return;
    }

    env->accd[reg] = datatype;
    memset(acc, 0, riscv_ame_acc_bytes(env));
}

void HELPER(ame_msettyp)(CPURISCVState *env, uint32_t md, uint32_t datatype)
{
    ame_require_enabled(env, GETPC());
    unsigned size = datatype & 0xff;
    unsigned group_size;

    if (md >= riscv_ame_mregs(env)) {
        riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, GETPC());
    }
    if (!ame_datatype_valid(datatype)) {
        env->amestatus |= AMESTATUS_UN;
        return;
    }

    group_size = MAX(1, DIV_ROUND_UP(size, riscv_ame_unit_bits(env)));
    if (md % group_size ||
        md + group_size > riscv_ame_mregs(env)) {
        riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, GETPC());
    }

    env->md[md] = datatype;
    memset(riscv_ame_mreg(env, md), 0, group_size * riscv_ame_mreg_bytes(env));
}

target_ulong HELPER(ame_acquire)(CPURISCVState *env,
                                 target_ulong descriptor)
{
    ame_require_ms(env, GETPC());
    (void)descriptor;

    if (!(env->ameown & AMEOWN_OWNED)) {
        memset(env->mreg, 0, sizeof(env->mreg));
        memset(env->md, 0, sizeof(env->md));
        memset(env->acc, 0, sizeof(env->acc));
        memset(env->accd, 0, sizeof(env->accd));
        env->amestatus = 0;
        env->amestype = 0;
        env->amefflags = 0;
        env->amexsat = 0;
        env->ameown = AMEOWN_OWNED;
#ifndef CONFIG_USER_ONLY
        /* A newly acquired context starts in the Initial state. */
        env->mstatus = (env->mstatus & ~MSTATUS_MS) | (1ULL << 23);
#endif
    }

    return 1;
}

void HELPER(ame_release)(CPURISCVState *env)
{
    ame_require_ms(env, GETPC());
    if (env->ameown & AMEOWN_OWNED) {
        env->ameown = 0;
#ifndef CONFIG_USER_ONLY
        /* Releasing the backend leaves the context in Initial state. */
        env->mstatus = (env->mstatus & ~MSTATUS_MS) | (1ULL << 23);
#endif
    }
}

void HELPER(ame_mzero_2d_acc)(CPURISCVState *env, uint32_t acc)
{
    ame_require_enabled(env, GETPC());
    if (acc >= riscv_ame_accs(env)) {
        riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, GETPC());
    }
    if (!ame_datatype_valid(env->accd[acc])) {
        env->amestatus |= AMESTATUS_UN;
        return;
    }
    memset(riscv_ame_acc(env, acc), 0, riscv_ame_acc_bytes(env));
}

typedef struct AMEMOperand {
    unsigned size;
    unsigned group_size;
    unsigned pack_factor;
    unsigned span;
} AMEMOperand;

static void ame_m_operand_init(CPURISCVState *env, AMEMOperand *op,
                                uint32_t datatype);
static bool ame_m_operand_form(CPURISCVState *env, AMEMOperand *op,
                               unsigned n_sq,
                               unsigned base);
static const uint8_t *ame_m_operand_square_buf(CPURISCVState *env,
                                                const uint8_t *mregs,
                                                unsigned base,
                                                unsigned square,
                                                unsigned size,
                                                unsigned *bit_offset);
static uint8_t *ame_m_operand_square_buf_mut(CPURISCVState *env,
                                             uint8_t *mregs, unsigned base,
                                             unsigned square, unsigned size,
                                             unsigned *bit_offset);

static bool ame_m_typed_operand(CPURISCVState *env, unsigned md,
                                AMEMOperand *op, uintptr_t ra)
{
    uint32_t datatype = env->md[md];

    if (!ame_datatype_valid(datatype)) {
        env->amestatus |= AMESTATUS_UN;
        return false;
    }
    ame_m_operand_init(env, op, datatype);
    if (!ame_m_operand_form(env, op, op->pack_factor, md)) {
        riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, ra);
    }
    return true;
}

static bool ame_acc_index_span(CPURISCVState *env, unsigned acc,
                               unsigned count, uintptr_t ra)
{
    if (count > riscv_ame_accs(env)) {
        env->amestatus |= AMESTATUS_UN;
        return false;
    }
    if (acc + count > riscv_ame_accs(env)) {
        riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, ra);
    }
    return true;
}

void HELPER(ame_mzero_2d_m)(CPURISCVState *env, uint32_t md)
{
    ame_require_enabled(env, GETPC());
    AMEMOperand op;

    if (md >= riscv_ame_mregs(env)) {
        riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, GETPC());
    }
    if (!ame_m_typed_operand(env, md, &op, GETPC())) {
        return;
    }
    memset(riscv_ame_mreg(env, md), 0, op.span * riscv_ame_mreg_bytes(env));
}

void HELPER(ame_mmov_ma)(CPURISCVState *env, uint32_t md, uint32_t acc)
{
    ame_require_enabled(env, GETPC());
    uint8_t result[AME_MAX_ACC_REG_BYTES];
    uint32_t dtype_m, dtype_acc;
    unsigned size, group_size, pack_factor;

    if (md >= riscv_ame_mregs(env) || acc >= riscv_ame_accs(env)) {
        riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, GETPC());
    }

    dtype_m = env->md[md];
    dtype_acc = env->accd[acc];
    if (!ame_datatype_valid(dtype_m) || !ame_datatype_valid(dtype_acc) ||
        dtype_m != dtype_acc) {
        env->amestatus |= AMESTATUS_UN;
        return;
    }

    size = dtype_m & 0xff;
    if (!size || size > AME_MAX_DATATYPE_SIZE ||
        (size < riscv_ame_unit_bits(env) &&
         riscv_ame_unit_bits(env) % size)) {
        env->amestatus |= AMESTATUS_UN;
        return;
    }

    group_size = ame_group_size(env, size);
    pack_factor = ame_pack_factor(env, size);
    if (!ame_acc_index_span(env, acc, pack_factor, GETPC())) {
        return;
    }
    if (md % group_size || md + group_size > riscv_ame_mregs(env)) {
        riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, GETPC());
    }

    if (pack_factor == 1) {
        memcpy(result, riscv_ame_acc(env, acc),
               group_size * riscv_ame_mreg_bytes(env));
    } else {
        unsigned square_bytes = riscv_ame_nelem(env) * size / 8;

        for (unsigned square = 0; square < pack_factor; square++) {
            memcpy(result + square * square_bytes,
                   riscv_ame_acc(env, acc + square),
                   square_bytes);
        }
    }
    memcpy(riscv_ame_mreg(env, md), result,
           group_size * riscv_ame_mreg_bytes(env));
}

void HELPER(ame_mmov_am)(CPURISCVState *env, uint32_t acc, uint32_t ms)
{
    ame_require_enabled(env, GETPC());
    uint32_t dtype_m, dtype_acc;
    unsigned size, group_size, pack_factor;
    uint8_t source[AME_MAX_M_FILE_BYTES];

    if (ms >= riscv_ame_mregs(env) || acc >= riscv_ame_accs(env)) {
        riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, GETPC());
    }
    dtype_m = env->md[ms];
    dtype_acc = env->accd[acc];
    if (!ame_datatype_valid(dtype_m) || !ame_datatype_valid(dtype_acc) ||
        dtype_m != dtype_acc) {
        env->amestatus |= AMESTATUS_UN;
        return;
    }
    size = dtype_m & 0xff;
    group_size = ame_group_size(env, size);
    pack_factor = ame_pack_factor(env, size);
    if (!ame_acc_index_span(env, acc, pack_factor, GETPC())) {
        return;
    }
    if (ms % group_size || ms + group_size > riscv_ame_mregs(env)) {
        riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, GETPC());
    }

    memcpy(source, env->mreg, riscv_ame_mfile_bytes(env));
    if (pack_factor == 1) {
        memcpy(riscv_ame_acc(env, acc), &source[ms * riscv_ame_mreg_bytes(env)],
               group_size * riscv_ame_mreg_bytes(env));
    } else {
        unsigned square_bytes = riscv_ame_nelem(env) * size / 8;

        for (unsigned square = 0; square < pack_factor; square++) {
            memcpy(riscv_ame_acc(env, acc + square),
                   source + ms * riscv_ame_mreg_bytes(env) +
                   square * square_bytes,
                   square_bytes);
        }
    }
}

void HELPER(ame_mmov_mm)(CPURISCVState *env, uint32_t md, uint32_t ms)
{
    ame_require_enabled(env, GETPC());
    AMEMOperand dest, source;
    uint8_t snapshot[AME_MAX_M_FILE_BYTES];

    if (md >= riscv_ame_mregs(env) || ms >= riscv_ame_mregs(env)) {
        riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, GETPC());
    }
    if (!ame_datatype_valid(env->md[md]) ||
        !ame_datatype_valid(env->md[ms]) || env->md[md] != env->md[ms]) {
        env->amestatus |= AMESTATUS_UN;
        return;
    }
    ame_m_operand_init(env, &dest, env->md[md]);
    ame_m_operand_init(env, &source, env->md[ms]);
    if (!ame_m_operand_form(env, &dest, dest.pack_factor, md) ||
        !ame_m_operand_form(env, &source, source.pack_factor, ms)) {
        riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, GETPC());
    }
    memcpy(snapshot, env->mreg, riscv_ame_mfile_bytes(env));
    memmove(riscv_ame_mreg(env, md), &snapshot[ms * riscv_ame_mreg_bytes(env)],
            source.span * riscv_ame_mreg_bytes(env));
}

static void ame_mwrite_id(CPURISCVState *env, uint32_t md, bool column,
                          uintptr_t ra)
{
    unsigned n = riscv_ame_n(env);
    AMEMOperand op;
    uint8_t result[AME_MAX_M_FILE_BYTES];
    uint32_t datatype;
    unsigned value_bits;

    if (md >= riscv_ame_mregs(env)) {
        riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, ra);
    }
    datatype = env->md[md];
    if (!ame_integer_datatype_valid(datatype)) {
        env->amestatus |= AMESTATUS_UN;
        return;
    }
    value_bits = (datatype & 0xff) - !!(datatype & BIT(30));
    if (value_bits < 32 && (uint32_t)(n - 1) >= (1u << value_bits)) {
        env->amestatus |= AMESTATUS_UN;
        return;
    }
    if (!ame_m_typed_operand(env, md, &op, ra)) {
        return;
    }
    memset(result, 0, riscv_ame_mfile_bytes(env));
    for (unsigned square = 0; square < op.pack_factor; square++) {
        unsigned offset;
        uint8_t *dst = ame_m_operand_square_buf_mut(env,
            result, md, square, op.size, &offset);

        for (unsigned row = 0; row < n; row++) {
            for (unsigned col = 0; col < n; col++) {
                unsigned elem = row * n + col;
                unsigned value = column ? col : row;

                ame_store_bits(dst, offset + elem * op.size, op.size,
                               value);
            }
        }
    }
    memcpy(riscv_ame_mreg(env, md), &result[md * riscv_ame_mreg_bytes(env)],
           op.span * riscv_ame_mreg_bytes(env));
}

void HELPER(ame_mcolid_ew)(CPURISCVState *env, uint32_t md)
{
    ame_require_enabled(env, GETPC());
    ame_mwrite_id(env, md, true, GETPC());
}

void HELPER(ame_mrowid_ew)(CPURISCVState *env, uint32_t md)
{
    ame_require_enabled(env, GETPC());
    ame_mwrite_id(env, md, false, GETPC());
}

static void ame_m_operand_init(CPURISCVState *env, AMEMOperand *op,
                                uint32_t datatype)
{
    op->size = datatype & 0xff;
    op->group_size = ame_group_size(env, op->size);
    op->pack_factor = ame_pack_factor(env, op->size);
    op->span = 0;
}

static bool ame_m_operand_form(CPURISCVState *env, AMEMOperand *op,
                               unsigned n_sq,
                               unsigned base)
{
    op->span = op->pack_factor > 1 ? n_sq / op->pack_factor :
                n_sq * op->group_size;
    return op->span != 0 && n_sq % op->pack_factor == 0 &&
           base % op->group_size == 0 &&
           base + op->span <= riscv_ame_mregs(env);
}

static unsigned ame_square_bit_offset(CPURISCVState *env, unsigned square,
                                       unsigned size)
{
    unsigned pack = ame_pack_factor(env, size);

    if (pack > 1) {
        unsigned reg = square / pack;
        unsigned slot = square % pack;

        return reg * riscv_ame_mreg_bytes(env) * 8 +
               slot * riscv_ame_nelem(env) * size;
    }
    return square * ame_group_size(env, size) * (riscv_ame_mreg_bytes(env) * 8);
}

static const uint8_t *ame_m_operand_square(CPURISCVState *env,
                                            unsigned base, unsigned square,
                                            unsigned size,
                                            unsigned *bit_offset)
{
    unsigned bit = ame_square_bit_offset(env, square, size);
    unsigned reg = bit / (riscv_ame_mreg_bytes(env) * 8);

    *bit_offset = bit % (riscv_ame_mreg_bytes(env) * 8);
    return riscv_ame_mreg(env, base + reg);
}

static const uint8_t *ame_m_operand_square_buf(CPURISCVState *env,
                                                const uint8_t *mregs,
                                                unsigned base,
                                                unsigned square,
                                                unsigned size,
                                                unsigned *bit_offset)
{
    unsigned bit = ame_square_bit_offset(env, square, size);
    unsigned reg = bit / (riscv_ame_mreg_bytes(env) * 8);

    *bit_offset = bit % (riscv_ame_mreg_bytes(env) * 8);
    return &mregs[(base + reg) * riscv_ame_mreg_bytes(env)];
}

static uint8_t *ame_m_operand_square_buf_mut(CPURISCVState *env,
                                             uint8_t *mregs, unsigned base,
                                             unsigned square, unsigned size,
                                             unsigned *bit_offset)
{
    return (uint8_t *)ame_m_operand_square_buf(env, mregs, base, square, size,
                                               bit_offset);
}

static __uint128_t ame_load_integer(const uint8_t *p, unsigned bit_offset,
                                    unsigned bits, bool is_signed)
{
    __uint128_t value = ame_load_bits(p, bit_offset, bits);

    if (is_signed && bits < 128 &&
        (value & ((__uint128_t)1 << (bits - 1)))) {
        value |= ~(((__uint128_t)1 << bits) - 1);
    }
    return value;
}

static float64 ame_load_float64(const uint8_t *p, unsigned bit_offset,
                                uint32_t datatype, float_status *fs)
{
    unsigned size = datatype & 0xff;
    uint64_t bits = (uint64_t)ame_load_bits(p, bit_offset, size);

    if (size == 16) {
        return ((datatype >> 26) & 0x1f) == 8 ?
               bfloat16_to_float64((bfloat16)bits, fs) :
               float16_to_float64((float16)bits, true, fs);
    }
    if (size == 32) {
        return float32_to_float64((float32)bits, fs);
    }
    return make_float64(bits);
}

static float128 ame_load_float128(const uint8_t *p, unsigned bit_offset,
                                  uint32_t datatype, float_status *fs)
{
    return float64_to_float128(ame_load_float64(p, bit_offset, datatype, fs),
                               fs);
}

static void ame_commit_float_flags(CPURISCVState *env, float_status *fs)
{
    FloatExceptionFlags flags = get_float_exception_flags(fs);

    env->amefflags |= ((flags & float_flag_invalid) ? BIT(4) : 0) |
                      ((flags & float_flag_divbyzero) ? BIT(3) : 0) |
                      ((flags & float_flag_overflow) ? BIT(2) : 0) |
                      ((flags & float_flag_underflow) ? BIT(1) : 0) |
                      ((flags & float_flag_inexact) ? BIT(0) : 0);
}

static __uint128_t ame_store_float128(uint8_t *p, unsigned bit_offset,
                                      uint32_t datatype, float128 value,
                                      float_status *fs)
{
    unsigned size = datatype & 0xff;
    uint64_t bits;

    if (size == 16) {
        bits = ((datatype >> 26) & 0x1f) == 8 ?
               float128_to_bfloat16(value, fs) :
               float128_to_float16(value, true, fs);
    } else if (size == 32) {
        bits = float128_to_float32(value, fs);
    } else {
        bits = float128_to_float64(value, fs);
    }
    ame_store_bits(p, bit_offset, size, bits);
    return bits;
}

/* Signed magnitude with a full 256-bit product; no signed host overflow. */
typedef struct AMEExactInteger {
    __uint128_t lo, hi;
    bool negative;
} AMEExactInteger;

static AMEExactInteger ame_exact_integer(__uint128_t bits, uint32_t dtype)
{
    unsigned size = dtype & 0xff;
    __uint128_t mask = size == 128 ? ~(__uint128_t)0 :
                      (((__uint128_t)1 << size) - 1);
    bool negative = (dtype & BIT(30)) &&
                    (bits & ((__uint128_t)1 << (size - 1)));

    bits &= mask;
    return (AMEExactInteger) { .lo = negative ? (-bits & mask) : bits,
                               .negative = negative && bits != 0 };
}

static int ame_exact_magnitude_compare(AMEExactInteger a, AMEExactInteger b)
{
    if (a.hi != b.hi) {
        return a.hi < b.hi ? -1 : 1;
    }
    return a.lo == b.lo ? 0 : a.lo < b.lo ? -1 : 1;
}

static AMEExactInteger ame_exact_add(AMEExactInteger a, AMEExactInteger b)
{
    AMEExactInteger r;

    if (a.negative == b.negative) {
        r.lo = a.lo + b.lo;
        r.hi = a.hi + b.hi + (r.lo < a.lo);
        r.negative = a.negative;
    } else {
        if (ame_exact_magnitude_compare(a, b) < 0) {
            AMEExactInteger temp = a;
            a = b;
            b = temp;
        }
        r.lo = a.lo - b.lo;
        r.hi = a.hi - b.hi - (a.lo < b.lo);
        r.negative = a.negative;
    }
    r.negative &= r.lo != 0 || r.hi != 0;
    return r;
}

static AMEExactInteger ame_exact_product(AMEExactInteger a, AMEExactInteger b)
{
    uint64_t av[2] = { a.lo, a.lo >> 64 };
    uint64_t bv[2] = { b.lo, b.lo >> 64 };
    uint64_t result[4] = { 0 };

    for (unsigned i = 0; i < 2; i++) {
        __uint128_t carry = 0;

        for (unsigned j = 0; j < 2; j++) {
            __uint128_t v = (__uint128_t)av[i] * bv[j] +
                            result[i + j] + carry;
            result[i + j] = v;
            carry = v >> 64;
        }
        result[i + 2] = carry;
    }
    return (AMEExactInteger) {
        .lo = ((__uint128_t)result[1] << 64) | result[0],
        .hi = ((__uint128_t)result[3] << 64) | result[2],
        .negative = a.negative != b.negative && a.lo != 0 && b.lo != 0,
    };
}

static __uint128_t ame_exact_convert(AMEExactInteger v, uint32_t dtype,
                                     bool *saturated)
{
    unsigned size = dtype & 0xff;
    bool sign = dtype & BIT(30);
    __uint128_t mask = size == 128 ? ~(__uint128_t)0 :
                      (((__uint128_t)1 << size) - 1);

    if (dtype & BIT(29)) {
        __uint128_t limit = sign ? ((__uint128_t)1 << (size - 1)) : mask;

        if (v.negative && !sign) {
            *saturated |= v.lo != 0 || v.hi != 0;
            return 0;
        }
        if (sign && !v.negative) {
            limit--;
        }
        if (v.hi || v.lo > limit) {
            v.lo = limit;
            *saturated = true;
        }
    }
    return (v.negative ? -v.lo : v.lo) & mask;
}

void HELPER(ame_mmulacc_2d)(CPURISCVState *env, uint32_t ms1,
                            uint32_t ms2, uint32_t acc, uint32_t funct7)
{
    unsigned n = riscv_ame_n(env);
    ame_require_enabled(env, GETPC());
    uint32_t dtype_a, dtype_b, dtype_c;
    unsigned size_c, n_sq;
    bool transpose_a, transpose_b, negate;
    AMEMOperand a, b;
    uint8_t result[AME_MAX_ACC_REG_BYTES];

    if (ms1 >= riscv_ame_mregs(env) || ms2 >= riscv_ame_mregs(env) ||
        acc >= riscv_ame_accs(env)) {
        riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, GETPC());
    }

    switch (funct7) {
    case AME_MMULACC:
    case AME_MMULACCNEG:
    case AME_MMULATACC:
    case AME_MMULATACCNEG:
    case AME_MMULBTACC:
    case AME_MMULBTACCNEG:
        break;
    default:
        env->amestatus |= AMESTATUS_UN;
        return;
    }
    transpose_a = funct7 == AME_MMULATACC || funct7 == AME_MMULATACCNEG;
    transpose_b = funct7 == AME_MMULBTACC || funct7 == AME_MMULBTACCNEG;
    negate = funct7 == AME_MMULACCNEG || funct7 == AME_MMULATACCNEG ||
             funct7 == AME_MMULBTACCNEG;

    dtype_a = env->md[ms1];
    dtype_b = env->md[ms2];
    dtype_c = env->accd[acc];
    if (!ame_datatype_valid(dtype_a) || !ame_datatype_valid(dtype_b) ||
        !ame_datatype_valid(dtype_c)) {
        env->amestatus |= AMESTATUS_UN;
        return;
    }

    if (((dtype_a ^ dtype_b) & (1u << 8)) ||
        ((dtype_a ^ dtype_c) & (1u << 8))) {
        env->amestatus |= AMESTATUS_UN;
        return;
    }

    size_c = dtype_c & 0xff;
    ame_m_operand_init(env, &a, dtype_a);
    ame_m_operand_init(env, &b, dtype_b);
    if (size_c < MAX(a.size, b.size)) {
        env->amestatus |= AMESTATUS_UN;
        return;
    }

    n_sq = MAX(a.pack_factor, b.pack_factor);
    if (!ame_m_operand_form(env, &a, n_sq, ms1) ||
        !ame_m_operand_form(env, &b, n_sq, ms2)) {
        riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, GETPC());
    }

    memcpy(result, riscv_ame_acc(env, acc), riscv_ame_acc_bytes(env));

    if ((dtype_a | dtype_b | dtype_c) & (1u << 8)) {
        float_status fs = env->fp_status;
        /* Accepted FP tuples use a deterministic binary128 non-fused fold;
         * the accumulator datatype is applied once after the complete fold. */
        set_float_rounding_mode(float_round_to_odd, &fs);
        set_float_exception_flags(0, &fs);
        set_default_nan_mode(true, &fs);
        set_float_detect_tininess(float_tininess_after_rounding, &fs);
        set_flush_to_zero(false, &fs);
        set_flush_inputs_to_zero(false, &fs);

        for (unsigned i = 0; i < n; i++) {
            for (unsigned j = 0; j < n; j++) {
                unsigned c_offset = (i * n + j) * size_c;
                float128 total = ame_load_float128(result, c_offset, dtype_c,
                                                   &fs);
                for (unsigned s = 0; s < n_sq; s++) {
                    unsigned a_offset, b_offset;
                    const uint8_t *a_square = ame_m_operand_square(
                        env, ms1, s, a.size, &a_offset);
                    const uint8_t *b_square = ame_m_operand_square(
                        env, ms2, s, b.size, &b_offset);
                    for (unsigned q = 0; q < n; q++) {
                        float128 av = ame_load_float128(a_square,
                            a_offset + ((transpose_a ? q * n + i :
                                         i * n + q) * a.size),
                            dtype_a, &fs);
                        float128 bv = ame_load_float128(b_square,
                            b_offset + ((transpose_b ? j * n + q :
                                         q * n + j) * b.size),
                            dtype_b, &fs);
                        float128 product = float128_mul(av, bv, &fs);
                        if (negate) {
                            product = float128_chs(product);
                        }
                        total = float128_add(total, product, &fs);
                    }
                }
                ame_float_rounding_from_dtype(&fs, dtype_c);
                ame_store_float128(result, c_offset, dtype_c, total, &fs);
                set_float_rounding_mode(float_round_to_odd, &fs);
            }
        }
        ame_commit_float_flags(env, &fs);
        memcpy(riscv_ame_acc(env, acc), result, riscv_ame_acc_bytes(env));
        return;
    }

    /* Fold in (square, inner) order. Each exact seed +/- product is
     * converted to C: clamp for Sat, otherwise wrap modulo C's width. */
    bool saturated = false;

    for (unsigned i = 0; i < n; i++) {
        for (unsigned j = 0; j < n; j++) {
            unsigned c_offset = (i * n + j) * size_c;
            __uint128_t sum = ame_load_bits(result, c_offset, size_c);
            bool saturate = dtype_c & (1u << 29);

            for (unsigned s = 0; s < n_sq; s++) {
                unsigned a_offset, b_offset;
                const uint8_t *a_square = ame_m_operand_square(
                    env, ms1, s, a.size, &a_offset);
                const uint8_t *b_square = ame_m_operand_square(
                    env, ms2, s, b.size, &b_offset);

                for (unsigned q = 0; q < n; q++) {
                    __uint128_t av = ame_load_integer(
                        a_square,
                        a_offset + ((transpose_a ? q * n + i :
                                     i * n + q) * a.size), a.size,
                        dtype_a & (1u << 30));
                    __uint128_t bv = ame_load_integer(
                        b_square,
                        b_offset + ((transpose_b ? j * n + q :
                                     q * n + j) * b.size), b.size,
                        dtype_b & (1u << 30));
                    __uint128_t product = av * bv;

                    if (saturate) {
                        AMEExactInteger p = ame_exact_product(
                            ame_exact_integer(av, dtype_a),
                            ame_exact_integer(bv, dtype_b));

                        p.negative ^= negate;
                        sum = ame_exact_convert(ame_exact_add(
                            ame_exact_integer(sum, dtype_c), p),
                            dtype_c, &saturated);
                        continue;
                    }

                    if (negate) {
                        product = -product;
                    }

                    if (size_c < 128) {
                        product &= (((__uint128_t)1 << size_c) - 1);
                    }
                    sum += product;
                    if (size_c < 128) {
                        sum &= (((__uint128_t)1 << size_c) - 1);
                    }
                }
            }
            ame_store_bits(result, c_offset, size_c, sum);
        }
    }
    memcpy(riscv_ame_acc(env, acc), result, riscv_ame_acc_bytes(env));
    if (saturated) {
        env->amexsat |= AMEXSAT_SAT;
    }
}

void HELPER(ame_mbitwise)(CPURISCVState *env, uint32_t md, uint32_t ms1,
                          uint32_t ms2, target_ulong scalar,
                          uint32_t funct7)
{
    ame_require_enabled(env, GETPC());
    uint32_t dtype_d, dtype_1, dtype_2;
    bool scalar_form = funct7 & 1;
    bool shift_op = funct7 >= AME_MSLL && funct7 <= AME_MSRL + 1;
    bool saturated = false;
    uint32_t operation = scalar_form ? funct7 - 1 : funct7;
    unsigned data_size, n_sq, span;
    AMEMOperand d, op1 = { 0 }, op2;
    uint8_t source[AME_MAX_M_FILE_BYTES];
    uint8_t result[AME_MAX_M_FILE_BYTES];
    __uint128_t scalar_bits = scalar;

    if (md >= riscv_ame_mregs(env) || ms1 >= riscv_ame_mregs(env) ||
        ms2 >= riscv_ame_mregs(env)) {
        riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, GETPC());
    }

    switch (operation) {
    case AME_MAND:
    case AME_MANDNOT:
    case AME_MOR:
    case AME_MORNOT:
    case AME_MSLL:
    case AME_MSRA:
    case AME_MSRL:
    case AME_MXOR:
        break;
    default:
        env->amestatus |= AMESTATUS_UN;
        return;
    }

    dtype_d = env->md[md];
    dtype_1 = scalar_form ? 0 : env->md[ms1];
    dtype_2 = env->md[ms2];
    if (!ame_integer_datatype_valid(dtype_d) ||
        !ame_integer_datatype_valid(dtype_2) ||
        (!scalar_form && !ame_integer_datatype_valid(dtype_1))) {
        env->amestatus |= AMESTATUS_UN;
        return;
    }

    if (!shift_op && (dtype_d != dtype_2 ||
                      (!scalar_form && dtype_d != dtype_1))) {
        env->amestatus |= AMESTATUS_UN;
        return;
    }
    if (operation == AME_MSRA &&
        !((scalar_form ? dtype_2 : dtype_1) & (1u << 30))) {
        env->amestatus |= AMESTATUS_UN;
        return;
    }

    /* The data scalar's type is part of the tuple, before group checks. */
    if (scalar_form && !shift_op &&
        !ame_integer_datatype_valid(env->amestype)) {
        env->amestatus |= AMESTATUS_UN;
        return;
    }

    ame_m_operand_init(env, &d, dtype_d);
    ame_m_operand_init(env, &op2, dtype_2);
    if (!scalar_form) {
        ame_m_operand_init(env, &op1, dtype_1);
    }
    n_sq = MAX(d.pack_factor, op2.pack_factor);
    if (!scalar_form) {
        n_sq = MAX(n_sq, op1.pack_factor);
    }
    if (!ame_m_operand_form(env, &d, n_sq, md) ||
        !ame_m_operand_form(env, &op2, n_sq, ms2) ||
        (!scalar_form && !ame_m_operand_form(env, &op1, n_sq, ms1))) {
        riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, GETPC());
    }

    /* Snapshot every source before writing a potentially overlapping group. */
    memcpy(source, env->mreg, riscv_ame_mfile_bytes(env));
    memcpy(result, env->mreg, riscv_ame_mfile_bytes(env));
    data_size = scalar_form ? op2.size : op1.size;
    if (scalar_form && !shift_op) {
        scalar_bits = ame_exact_convert(
            ame_exact_integer(scalar, env->amestype), dtype_2, &saturated);
    }

    for (unsigned square = 0; square < n_sq; square++) {
        unsigned d_offset, a_offset = 0, b_offset;
        uint8_t *d_square = (uint8_t *)ame_m_operand_square_buf(env,
            result, md, square, d.size, &d_offset);
        const uint8_t *a_square = scalar_form ? NULL :
            ame_m_operand_square_buf(env, source, ms1, square, op1.size,
                                     &a_offset);
        const uint8_t *b_square = ame_m_operand_square_buf(env,
            source, ms2, square, op2.size, &b_offset);

        for (unsigned elem = 0; elem < riscv_ame_nelem(env); elem++) {
            __uint128_t av = scalar_form ? 0 :
                ame_load_bits(a_square, a_offset + elem * op1.size,
                              op1.size);
            __uint128_t bv = ame_load_bits(b_square,
                                           b_offset + elem * op2.size,
                                           op2.size);
            __uint128_t value;

            switch (operation) {
            case AME_MAND:
                value = (scalar_form ? scalar_bits : av) & bv;
                break;
            case AME_MANDNOT:
                value = (scalar_form ? scalar_bits : av) & ~bv;
                break;
            case AME_MOR:
                value = (scalar_form ? scalar_bits : av) | bv;
                break;
            case AME_MORNOT:
                value = (scalar_form ? scalar_bits : av) | ~bv;
                break;
            case AME_MXOR:
                value = (scalar_form ? scalar_bits : av) ^ bv;
                break;
            case AME_MSLL: {
                unsigned shift = (unsigned)((scalar_form ? scalar : bv) %
                                            data_size);
                AMEExactInteger data = ame_exact_integer(
                    scalar_form ? bv : av, scalar_form ? dtype_2 : dtype_1);

                /* Preserve up to 255 result bits before final conversion. */
                data.hi = shift ? data.lo >> (128 - shift) : 0;
                data.lo <<= shift;
                value = ame_exact_convert(data, dtype_d, &saturated);
                break;
            }
            case AME_MSRA: {
                unsigned shift = (unsigned)((scalar_form ? scalar : bv) %
                                            data_size);
                const uint8_t *data_square = scalar_form ? b_square :
                                             a_square;
                unsigned data_offset = scalar_form ? b_offset : a_offset;
                __int128_t data = (__int128_t)ame_load_integer(
                    data_square, data_offset + elem * data_size,
                    data_size, true);

                value = ame_exact_convert(
                    ame_exact_integer((__uint128_t)(data >> shift),
                                      scalar_form ? dtype_2 : dtype_1),
                    dtype_d, &saturated);
                break;
            }
            case AME_MSRL: {
                unsigned shift = (unsigned)((scalar_form ? scalar : bv) %
                                            data_size);
                /* Logical right shift produces an unsigned integer value. */
                AMEExactInteger data = {
                    .lo = (scalar_form ? bv : av) >> shift,
                };

                value = ame_exact_convert(data, dtype_d, &saturated);
                break;
            }
            default:
                g_assert_not_reached();
            }
            ame_store_bits(d_square, d_offset + elem * d.size, d.size,
                           value);
        }
    }

    span = d.span * riscv_ame_mreg_bytes(env);
    memcpy(riscv_ame_mreg(env, md),
           result + md * riscv_ame_mreg_bytes(env), span);
    if (saturated) {
        env->amexsat |= AMEXSAT_SAT;
    }
}

static int ame_compare_integer_values(__uint128_t av, bool signed_a,
                                      __uint128_t bv, bool signed_b)
{
    bool negative_a = signed_a && (__int128_t)av < 0;
    bool negative_b = signed_b && (__int128_t)bv < 0;

    if (negative_a != negative_b) {
        return negative_a ? -1 : 1;
    }
    if (negative_a) {
        return (__int128_t)av < (__int128_t)bv ? -1 :
               (__int128_t)av > (__int128_t)bv ? 1 : 0;
    }
    return av < bv ? -1 : av > bv ? 1 : 0;
}

static int ame_compare_elements(const uint8_t *a, unsigned a_offset,
                                uint32_t dtype_a, const uint8_t *b,
                                unsigned b_offset, uint32_t dtype_b,
                                float_status *fs)
{
    if (dtype_a & (1u << 8)) {
        FloatRelation relation = float64_compare_quiet(
            ame_load_float64(a, a_offset, dtype_a, fs),
            ame_load_float64(b, b_offset, dtype_b, fs), fs);

        return relation == float_relation_unordered ? 2 : relation;
    }
    return ame_compare_integer_values(
        ame_load_integer(a, a_offset, dtype_a & 0xff,
                         dtype_a & (1u << 30)),
        dtype_a & (1u << 30),
        ame_load_integer(b, b_offset, dtype_b & 0xff,
                         dtype_b & (1u << 30)),
        dtype_b & (1u << 30));
}

void HELPER(ame_mcompare_select)(CPURISCVState *env, uint32_t md,
                                 uint32_t ms1, uint32_t ms2,
                                 target_ulong scalar, uint32_t funct7)
{
    ame_require_enabled(env, GETPC());
    bool scalar_form = funct7 == AME_MCMPGE_X || funct7 == AME_MCMPLT_X;
    bool compare_result = funct7 >= AME_MCMPGE && funct7 <= AME_MCMPLT_X;
    bool test_less = funct7 == AME_MCMOVLT || funct7 == AME_MCMPLT ||
                     funct7 == AME_MCMPLT_X || funct7 == AME_MSELLT;
    bool preserve_false = funct7 == AME_MCMOVGE || funct7 == AME_MCMOVLT;
    uint32_t dtype_d, dtype_a, dtype_b;
    unsigned n_sq, span;
    AMEMOperand d, a = { 0 }, b;
    uint8_t source[AME_MAX_M_FILE_BYTES];
    uint8_t result[AME_MAX_M_FILE_BYTES];
    uint8_t scalar_value[sizeof(__uint128_t)] = { 0 };
    const uint8_t zero_value[sizeof(uint64_t)] = { 0 };
    float_status fs = env->fp_status;

    if (md >= riscv_ame_mregs(env) || ms1 >= riscv_ame_mregs(env) ||
        ms2 >= riscv_ame_mregs(env) || funct7 < AME_MCMOVGE ||
        funct7 > AME_MSELLT) {
        riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, GETPC());
    }

    dtype_d = env->md[md];
    dtype_a = scalar_form ? env->amestype : env->md[ms1];
    dtype_b = env->md[ms2];
    if (!ame_datatype_valid(dtype_a) || !ame_datatype_valid(dtype_b) ||
        (compare_result ? !ame_integer_datatype_valid(dtype_d) :
                          !ame_datatype_valid(dtype_d))) {
        env->amestatus |= AMESTATUS_UN;
        return;
    }
    if ((!compare_result && dtype_d != dtype_b) ||
        (compare_result && ((dtype_a ^ dtype_b) & (1u << 8))) ||
        (scalar_form && dtype_a != dtype_b)) {
        env->amestatus |= AMESTATUS_UN;
        return;
    }

    ame_m_operand_init(env, &d, dtype_d);
    ame_m_operand_init(env, &b, dtype_b);
    if (!scalar_form) {
        ame_m_operand_init(env, &a, dtype_a);
    }
    n_sq = MAX(d.pack_factor, b.pack_factor);
    if (!scalar_form) {
        n_sq = MAX(n_sq, a.pack_factor);
    }
    if (!ame_m_operand_form(env, &d, n_sq, md) ||
        !ame_m_operand_form(env, &b, n_sq, ms2) ||
        (!scalar_form && !ame_m_operand_form(env, &a, n_sq, ms1))) {
        riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, GETPC());
    }

    memcpy(source, env->mreg, riscv_ame_mfile_bytes(env));
    memcpy(result, env->mreg, riscv_ame_mfile_bytes(env));
    if (scalar_form) {
        ame_store_bits(scalar_value, 0, dtype_a & 0xff, scalar);
    }
    set_float_exception_flags(0, &fs);

    for (unsigned square = 0; square < n_sq; square++) {
        unsigned d_offset, a_offset = 0, b_offset;
        uint8_t *d_square = (uint8_t *)ame_m_operand_square_buf(env,
            result, md, square, d.size, &d_offset);
        const uint8_t *a_square = scalar_form ? scalar_value :
            ame_m_operand_square_buf(env, source, ms1, square, a.size,
                                     &a_offset);
        const uint8_t *b_square = ame_m_operand_square_buf(env,
            source, ms2, square, b.size, &b_offset);

        for (unsigned elem = 0; elem < riscv_ame_nelem(env); elem++) {
            unsigned ae = scalar_form ? 0 : a_offset + elem * a.size;
            unsigned be = b_offset + elem * b.size;
            int relation;
            bool predicate;
            __uint128_t value;

            if (compare_result) {
                relation = ame_compare_elements(a_square, ae, dtype_a,
                                                b_square, be, dtype_b, &fs);
            } else if (dtype_a & (1u << 8)) {
                relation = ame_compare_elements(
                    a_square, ae, dtype_a, zero_value, 0, dtype_a, &fs);
            } else {
                __uint128_t av = ame_load_integer(
                    a_square, ae, a.size, dtype_a & (1u << 30));
                relation = (dtype_a & (1u << 30)) && (__int128_t)av < 0 ?
                           -1 : av != 0;
            }
            predicate = test_less ? relation == float_relation_less :
                        relation == float_relation_equal ||
                        relation == float_relation_greater;

            if (compare_result) {
                value = predicate ? (d.size == 128 ?
                                     ~(__uint128_t)0 :
                                     (((__uint128_t)1 << d.size) - 1)) : 0;
            } else if (predicate) {
                value = ame_load_bits(b_square, be, b.size);
            } else if (preserve_false) {
                value = ame_load_bits(d_square, d_offset + elem * d.size,
                                      d.size);
            } else {
                value = 0;
            }
            ame_store_bits(d_square, d_offset + elem * d.size, d.size,
                           value);
        }
    }

    span = d.span * riscv_ame_mreg_bytes(env);
    memcpy(riscv_ame_mreg(env, md),
           result + md * riscv_ame_mreg_bytes(env), span);
    ame_commit_float_flags(env, &fs);
}

static bool ame_dtype_is_float(uint32_t datatype)
{
    return datatype & (1u << 8);
}

static Int128 ame_int128_from_bits(__uint128_t bits)
{
    return int128_make128(bits, bits >> 64);
}

static __uint128_t ame_int128_bits(Int128 value)
{
    return ((__uint128_t)(uint64_t)int128_gethi(value) << 64) |
           int128_getlo(value);
}

static float128 ame_integer_to_float128(__uint128_t bits,
                                        uint32_t datatype,
                                        float_status *fs)
{
    unsigned size = datatype & 0xff;
    __uint128_t mask = size == 128 ? ~(__uint128_t)0 :
                       (((__uint128_t)1 << size) - 1);

    bits &= mask;
    if (datatype & BIT(30)) {
        if (size < 128 && (bits & ((__uint128_t)1 << (size - 1)))) {
            bits |= ~mask;
        }
        return int128_to_float128(ame_int128_from_bits(bits), fs);
    }
    return uint128_to_float128(ame_int128_from_bits(bits), fs);
}

static __uint128_t ame_integer_limits(unsigned size, bool is_signed,
                                      bool upper)
{
    if (is_signed) {
        if (upper) {
            return size == 128 ? (((__uint128_t)1 << 127) - 1) :
                   (((__uint128_t)1 << (size - 1)) - 1);
        }
        return size == 128 ? ((__uint128_t)1 << 127) :
               ((__uint128_t)1 << (size - 1));
    }
    return upper ? (size == 128 ? ~(__uint128_t)0 :
                    (((__uint128_t)1 << size) - 1)) : 0;
}

static __uint128_t ame_convert_integer(CPURISCVState *env,
                                       __uint128_t bits,
                                       uint32_t src_dtype,
                                       uint32_t dst_dtype)
{
    unsigned src_size = src_dtype & 0xff;
    unsigned dst_size = dst_dtype & 0xff;
    bool signed_src = src_dtype & BIT(30);
    bool signed_dst = dst_dtype & BIT(30);
    bool saturate = dst_dtype & BIT(29);
    __uint128_t src_mask = src_size == 128 ? ~(__uint128_t)0 :
                           (((__uint128_t)1 << src_size) - 1);
    __uint128_t dst_mask = dst_size == 128 ? ~(__uint128_t)0 :
                           (((__uint128_t)1 << dst_size) - 1);
    bool negative = signed_src && (bits & ((__uint128_t)1 <<
                                           (src_size - 1)));
    __uint128_t value_bits;
    bool overflow = false;

    bits &= src_mask;
    value_bits = negative ? bits | ~src_mask : bits;
    if (!saturate) {
        return value_bits & dst_mask;
    }

    if (signed_dst) {
        __uint128_t maximum = ame_integer_limits(dst_size, true, true);
        __int128_t minimum = dst_size == 128 ?
            (__int128_t)((__uint128_t)1 << 127) :
            -((__int128_t)1 << (dst_size - 1));

        if (signed_src) {
            __int128_t value = (__int128_t)value_bits;

            if (value < minimum) {
                bits = (__uint128_t)minimum;
                overflow = true;
            } else if (value > (__int128_t)maximum) {
                bits = maximum;
                overflow = true;
            } else {
                bits = value_bits;
            }
        } else if (bits > maximum) {
            bits = maximum;
            overflow = true;
        }
    } else {
        __uint128_t maximum = ame_integer_limits(dst_size, false, true);

        if (negative) {
            bits = 0;
            overflow = true;
        } else if (bits > maximum) {
            bits = maximum;
            overflow = true;
        }
    }

    if (overflow) {
        env->amexsat |= AMEXSAT_SAT;
    }
    return bits & dst_mask;
}

static float128 ame_float_integer_limit(unsigned size, bool signed_type,
                                        bool upper, float_status *fs)
{
    float128 one = int64_to_float128(1, fs);
    float128 limit;

    if (size < 128) {
        if (signed_type) {
            __int128_t value = upper ?
                (((__int128_t)1 << (size - 1)) - 1) :
                -((__int128_t)1 << (size - 1));
            return int128_to_float128(ame_int128_from_bits(value), fs);
        }
        if (upper) {
            uint64_t value = ame_integer_limits(size, false, true);
            return uint64_to_float128(value, fs);
        }
        return float128_zero;
    }

    limit = float128_scalbn(one, 127, fs);
    if (signed_type) {
        return upper ? limit : float128_chs(limit);
    }
    return upper ? float128_scalbn(one, 128, fs) : float128_zero;
}

static __uint128_t ame_convert_float_to_integer(CPURISCVState *env,
                                                float128 value,
                                                uint32_t dst_dtype,
                                                float_status *fs)
{
    unsigned size = dst_dtype & 0xff;
    bool signed_type = dst_dtype & BIT(30);
    bool saturate = dst_dtype & BIT(29);
    __uint128_t mask = size == 128 ? ~(__uint128_t)0 :
                       (((__uint128_t)1 << size) - 1);
    float_status round_fs = *fs;
    float_status bound_fs = *fs;
    float128 rounded, minimum, maximum;
    bool out_of_range;
    unsigned rounding = (dst_dtype >> 27) & 3;

    set_float_exception_flags(0, &round_fs);
    if (rounding == 0) {
        /* Round the original value so NX reflects its discarded fraction. */
        set_float_rounding_mode(float_round_to_zero, &round_fs);
        rounded = float128_round_to_int(value, &round_fs);
        if (get_float_exception_flags(&round_fs) & float_flag_inexact) {
            float128 one = int64_to_float128(1, &round_fs);
            float128 half = float128_scalbn(one, -1, &round_fs);
            float128 fraction = float128_abs(float128_sub(value, rounded,
                                                          &round_fs));
            FloatRelation relation = float128_compare_quiet(fraction, half,
                                                            &round_fs);

            if (relation == float_relation_greater ||
                (relation == float_relation_equal && !float128_is_neg(value))) {
                rounded = float128_add(rounded, float128_is_neg(value) ?
                                       float128_chs(one) : one, &round_fs);
            }
        }
    } else {
        static const FloatRoundMode modes[] = {
            float_round_nearest_even, float_round_down, float_round_to_odd,
        };

        set_float_rounding_mode(modes[rounding - 1], &round_fs);
        rounded = float128_round_to_int(value, &round_fs);
    }
    float_raise(get_float_exception_flags(&round_fs), fs);

    if (float128_is_any_nan(value)) {
        float_raise(float_flag_invalid, fs);
        return 0;
    }
    if (float128_is_infinity(value)) {
        float_raise(float_flag_invalid | float_flag_invalid_cvti, fs);
        if (!saturate) {
            return signed_type && float128_is_neg(value) ?
                   ((__uint128_t)1 << (size - 1)) : mask;
        }
        env->amexsat |= AMEXSAT_SAT;
        return ame_integer_limits(size, signed_type, !float128_is_neg(value));
    }

    set_float_exception_flags(0, &bound_fs);
    minimum = ame_float_integer_limit(size, signed_type, false, &bound_fs);
    maximum = ame_float_integer_limit(size, signed_type, true, &bound_fs);
    if (signed_type) {
        out_of_range = float128_compare(rounded, minimum, &bound_fs) < 0 ||
                       (size == 128 ?
                        float128_compare(rounded, maximum, &bound_fs) >= 0 :
                        float128_compare(rounded, maximum, &bound_fs) > 0);
    } else {
        out_of_range = float128_compare(rounded, minimum, &bound_fs) < 0 ||
                       (size == 128 ?
                        float128_compare(rounded, maximum, &bound_fs) >= 0 :
                        float128_compare(rounded, maximum, &bound_fs) > 0);
    }

    if (out_of_range && saturate) {
        float_raise(float_flag_invalid | float_flag_invalid_cvti, fs);
        env->amexsat |= AMEXSAT_SAT;
        return float128_is_neg(rounded) ?
               (signed_type ? (((__uint128_t)1 << (size - 1)) & mask) : 0) :
               (signed_type ? (mask >> 1) : mask);
    }

    if (signed_type) {
        Int128 integer = float128_to_int128_round_to_zero(rounded, fs);
        return ame_int128_bits(integer) & mask;
    }
    {
        Int128 integer = float128_to_uint128_round_to_zero(rounded, fs);
        return ame_int128_bits(integer) & mask;
    }
}

static __uint128_t ame_convert_scalar(CPURISCVState *env,
                                      __uint128_t bits,
                                      uint32_t src_dtype,
                                      uint32_t dst_dtype,
                                      float_status *fs)
{
    bool src_float = ame_dtype_is_float(src_dtype);
    bool dst_float = ame_dtype_is_float(dst_dtype);
    float128 value;
    float_status source_fs = *fs;

    set_float_exception_flags(0, &source_fs);
    set_float_rounding_mode(float_round_to_odd, &source_fs);

    if (!src_float && !dst_float) {
        return ame_convert_integer(env, bits, src_dtype, dst_dtype);
    }

    if (src_float) {
        uint8_t source[sizeof(__uint128_t)] = { 0 };

        ame_store_bits(source, 0, src_dtype & 0xff, bits);
        value = ame_load_float128(source, 0, src_dtype, &source_fs);
    } else {
        value = ame_integer_to_float128(bits, src_dtype, &source_fs);
    }
    float_raise(get_float_exception_flags(&source_fs), fs);

    if (dst_float) {
        uint8_t result[sizeof(__uint128_t)] = { 0 };

        ame_float_rounding_from_dtype(fs, dst_dtype);
        ame_store_float128(result, 0, dst_dtype, value, fs);
        return ame_load_bits(result, 0, dst_dtype & 0xff);
    }
    return ame_convert_float_to_integer(env, value, dst_dtype, fs);
}

void HELPER(ame_mbcast)(CPURISCVState *env, uint32_t md,
                        target_ulong scalar, target_ulong dtype_value)
{
    ame_require_enabled(env, GETPC());
    uint32_t dst_dtype, src_dtype = (uint32_t)dtype_value;
    AMEMOperand dest;
    uint8_t result[AME_MAX_M_FILE_BYTES];
    uint8_t scalar_bits[sizeof(__uint128_t)] = { 0 };
    float_status fs = env->fp_status;

    if (md >= riscv_ame_mregs(env)) {
        riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, GETPC());
    }
    dst_dtype = env->md[md];
    if (!ame_conversion_type_supported(dst_dtype) ||
        !ame_conversion_type_supported(src_dtype)) {
        env->amestatus |= AMESTATUS_UN;
        return;
    }
    ame_m_operand_init(env, &dest, dst_dtype);
    if (!ame_m_operand_form(env, &dest, dest.pack_factor, md)) {
        riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, GETPC());
    }
    ame_store_bits(scalar_bits, 0, src_dtype & 0xff, scalar);
    set_float_exception_flags(0, &fs);
    __uint128_t converted = ame_convert_scalar(
        env, ame_load_bits(scalar_bits, 0, src_dtype & 0xff),
        src_dtype, dst_dtype, &fs);
    memcpy(result, env->mreg, riscv_ame_mfile_bytes(env));

    for (unsigned square = 0; square < dest.pack_factor; square++) {
        unsigned offset;
        uint8_t *dst = ame_m_operand_square_buf_mut(env,
            result, md, square, dest.size, &offset);

        for (unsigned elem = 0; elem < riscv_ame_nelem(env); elem++) {
            ame_store_bits(dst, offset + elem * dest.size, dest.size,
                           converted);
        }
    }
    memcpy(riscv_ame_mreg(env, md), &result[md * riscv_ame_mreg_bytes(env)],
           dest.span * riscv_ame_mreg_bytes(env));
    ame_commit_float_flags(env, &fs);
}

void HELPER(ame_mconv)(CPURISCVState *env, uint32_t md, uint32_t ms)
{
    ame_require_enabled(env, GETPC());
    uint32_t dst_dtype, src_dtype;
    unsigned n_sq;
    AMEMOperand dest, source;
    uint8_t snapshot[AME_MAX_M_FILE_BYTES];
    uint8_t result[AME_MAX_M_FILE_BYTES];
    float_status fs = env->fp_status;

    if (md >= riscv_ame_mregs(env) || ms >= riscv_ame_mregs(env)) {
        riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, GETPC());
    }
    dst_dtype = env->md[md];
    src_dtype = env->md[ms];
    if (!ame_conversion_type_supported(dst_dtype) ||
        !ame_conversion_type_supported(src_dtype)) {
        env->amestatus |= AMESTATUS_UN;
        return;
    }
    ame_m_operand_init(env, &dest, dst_dtype);
    ame_m_operand_init(env, &source, src_dtype);
    n_sq = MAX(dest.pack_factor, source.pack_factor);
    if (!ame_m_operand_form(env, &dest, n_sq, md) ||
        !ame_m_operand_form(env, &source, n_sq, ms)) {
        riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, GETPC());
    }
    memcpy(snapshot, env->mreg, riscv_ame_mfile_bytes(env));
    memcpy(result, snapshot, riscv_ame_mfile_bytes(env));
    set_float_exception_flags(0, &fs);

    for (unsigned square = 0; square < n_sq; square++) {
        unsigned source_offset, dest_offset;
        const uint8_t *src = ame_m_operand_square_buf(env,
            snapshot, ms, square, source.size, &source_offset);
        uint8_t *dst = ame_m_operand_square_buf_mut(env,
            result, md, square, dest.size, &dest_offset);

        for (unsigned elem = 0; elem < riscv_ame_nelem(env); elem++) {
            __uint128_t value = ame_convert_scalar(
                env, ame_load_bits(src, source_offset + elem * source.size,
                                   source.size),
                src_dtype, dst_dtype, &fs);
            ame_store_bits(dst, dest_offset + elem * dest.size, dest.size,
                           value);
        }
    }
    memcpy(riscv_ame_mreg(env, md), &result[md * riscv_ame_mreg_bytes(env)],
           dest.span * riscv_ame_mreg_bytes(env));
    ame_commit_float_flags(env, &fs);
}

void HELPER(ame_mmove_mx)(CPURISCVState *env, uint32_t md,
                          target_ulong pos, target_ulong scalar,
                          uint32_t size)
{
    ame_require_enabled(env, GETPC());
    unsigned elements;

    if ((size == 64 && TARGET_LONG_BITS < 64) ||
        size > riscv_ame_n(env) * riscv_ame_unit_bits(env)) {
        riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, GETPC());
    }
    elements = (riscv_ame_mreg_bytes(env) * 8) / size;
    pos &= elements - 1;
    ame_store_bits(riscv_ame_mreg(env, md), pos * size, size, scalar);
}

target_ulong HELPER(ame_mmove_xm)(CPURISCVState *env, uint32_t ms,
                                  target_ulong pos, uint32_t size)
{
    ame_require_enabled(env, GETPC());
    unsigned elements;

    if ((size == 64 && TARGET_LONG_BITS < 64) ||
        size > riscv_ame_n(env) * riscv_ame_unit_bits(env)) {
        riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, GETPC());
    }
    elements = (riscv_ame_mreg_bytes(env) * 8) / size;
    pos &= elements - 1;
    return (target_ulong)ame_load_bits(riscv_ame_mreg(env, ms),
                                       pos * size, size);
}

static void ame_mpack_common(CPURISCVState *env, uint32_t md,
                             target_ulong slot_value, uint32_t ms,
                             bool unpack, uintptr_t ra)
{
    uint32_t dst_dtype = env->md[md];
    uint32_t src_dtype = env->md[ms];
    unsigned slot;
    AMEMOperand dest, source;
    uint8_t snapshot[AME_MAX_M_FILE_BYTES];
    uint8_t result[AME_MAX_M_FILE_BYTES];
    float_status fs = env->fp_status;

    if (!ame_conversion_type_supported(dst_dtype) ||
        !ame_conversion_type_supported(src_dtype)) {
        env->amestatus |= AMESTATUS_UN;
        return;
    }
    ame_m_operand_init(env, &dest, dst_dtype);
    ame_m_operand_init(env, &source, src_dtype);
    if ((!unpack && (dest.pack_factor <= 1 || source.pack_factor != 1)) ||
        (unpack && (source.pack_factor <= 1 || dest.pack_factor != 1))) {
        env->amestatus |= AMESTATUS_UN;
        return;
    }
    slot = slot_value & ((unpack ? source.pack_factor : dest.pack_factor) - 1);
    if (!ame_m_operand_form(env, &dest, unpack ? 1 : dest.pack_factor, md) ||
        !ame_m_operand_form(env, &source,
                            unpack ? source.pack_factor : 1, ms)) {
        riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, ra);
    }
    memcpy(snapshot, env->mreg, riscv_ame_mfile_bytes(env));
    memcpy(result, snapshot, riscv_ame_mfile_bytes(env));
    unsigned source_square = unpack ? slot : 0;
    unsigned dest_square = unpack ? 0 : slot;
    unsigned source_offset, dest_offset;
    const uint8_t *src = ame_m_operand_square_buf(env,
        snapshot, ms, source_square, source.size, &source_offset);
    uint8_t *dst = ame_m_operand_square_buf_mut(env,
        result, md, dest_square, dest.size, &dest_offset);

    set_float_exception_flags(0, &fs);
    for (unsigned elem = 0; elem < riscv_ame_nelem(env); elem++) {
        __uint128_t value = ame_convert_scalar(
            env, ame_load_bits(src, source_offset + elem * source.size,
                               source.size),
            src_dtype, dst_dtype, &fs);
        ame_store_bits(dst, dest_offset + elem * dest.size, dest.size, value);
    }
    memcpy(riscv_ame_mreg(env, md), &result[md * riscv_ame_mreg_bytes(env)],
           dest.span * riscv_ame_mreg_bytes(env));
    ame_commit_float_flags(env, &fs);
}

void HELPER(ame_mpack)(CPURISCVState *env, uint32_t md,
                       target_ulong slot, uint32_t ms)
{
    ame_require_enabled(env, GETPC());
    ame_mpack_common(env, md, slot, ms, false, GETPC());
}

void HELPER(ame_munpack)(CPURISCVState *env, uint32_t md,
                         target_ulong slot, uint32_t ms)
{
    ame_require_enabled(env, GETPC());
    ame_mpack_common(env, md, slot, ms, true, GETPC());
}

static bool ame_rowcol_operand_type(CPURISCVState *env, unsigned reg,
                                    AMEMOperand *op)
{
    if (!ame_datatype_valid(env->md[reg])) {
        env->amestatus |= AMESTATUS_UN;
        return false;
    }
    ame_m_operand_init(env, op, env->md[reg]);
    if (op->pack_factor != 1) {
        env->amestatus |= AMESTATUS_UN;
        return false;
    }
    return true;
}

static void ame_rowcol_operand_form(CPURISCVState *env, unsigned reg,
                                    AMEMOperand *op, uintptr_t ra)
{
    if (!ame_m_operand_form(env, op, 1, reg)) {
        riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, ra);
    }
}

static __uint128_t ame_integer_add(CPURISCVState *env, __uint128_t a,
                                   __uint128_t b, uint32_t datatype)
{
    unsigned size = datatype & 0xff;
    bool is_signed = datatype & BIT(30);
    bool saturate = datatype & BIT(29);
    __uint128_t mask = size == 128 ? ~(__uint128_t)0 :
                       (((__uint128_t)1 << size) - 1);
    __uint128_t result = (a + b) & mask;
    bool overflow;

    a &= mask;
    b &= mask;
    if (!saturate) {
        return result;
    }

    if (is_signed) {
        __uint128_t sign = (__uint128_t)1 << (size - 1);

        overflow = !((a ^ b) & sign) && ((a ^ result) & sign);
        if (overflow) {
            result = (a & sign) ? sign : sign - 1;
        }
    } else {
        overflow = size == 128 ? result < a : a + b > mask;
        if (overflow) {
            result = mask;
        }
    }
    if (overflow) {
        env->amexsat |= AMEXSAT_SAT;
    }
    return result;
}

static __uint128_t ame_float_default_nan(uint32_t datatype,
                                         float_status *fs)
{
    unsigned size = datatype & 0xff;

    if (size == 16) {
        return ((datatype >> 26) & 0x1f) == 8 ?
               bfloat16_default_nan(fs) : float16_default_nan(fs);
    }
    if (size == 32) {
        return float32_default_nan(fs);
    }
    return float64_default_nan(fs);
}

static void ame_float_rounding_from_dtype(float_status *fs,
                                          uint32_t datatype)
{
    static const FloatRoundMode modes[] = {
        float_round_nearest_even,
        float_round_to_zero,
        float_round_down,
        float_round_up,
        float_round_ties_away,
        float_round_to_odd,
    };

    set_float_rounding_mode(modes[(datatype >> 22) & 0xf], fs);
}

/*
 * Prefix/reduction tuples: all standard integer pairs, or pairs drawn from
 * IEEE FP16/BF16/FP32/FP64 with infinity and gradual underflow enabled.
 * Cross-class, custom and finite-only/flush-to-zero FP tuples are unsupported.
 */
static bool ame_reduction_float_type(uint32_t datatype)
{
    return ame_float_datatype_valid(datatype) &&
           (datatype & (BIT(21) | BIT(20))) == (BIT(21) | BIT(20));
}

static __uint128_t ame_reduction_float_convert(const uint8_t *src,
                                              unsigned offset,
                                              uint32_t src_type,
                                              uint32_t dst_type,
                                              float_status *fs)
{
    /* Widening to binary64 is exact; only the destination conversion rounds. */
    float64 value = ame_load_float64(src, offset, src_type, fs);
    unsigned size = dst_type & 0xff;

    if (float64_is_any_nan(value)) {
        if (float64_is_signaling_nan(value, fs)) {
            float_raise(float_flag_invalid, fs);
        }
        return ame_float_default_nan(dst_type, fs);
    }
    if (size == 16) {
        return ((dst_type >> 26) & 0x1f) == 8 ?
               float64_to_bfloat16(value, fs) :
               float64_to_float16(value, true, fs);
    }
    return size == 32 ? float64_to_float32(value, fs) : value;
}

static __uint128_t ame_reduction_float_add(__uint128_t a, __uint128_t b,
                                          uint32_t datatype,
                                          float_status *fs)
{
    /* Each association-tree node rounds directly at destination precision. */
    switch (datatype & 0xff) {
    case 16:
        return ((datatype >> 26) & 0x1f) == 8 ?
               bfloat16_add(a, b, fs) : float16_add(a, b, fs);
    case 32:
        return float32_add(a, b, fs);
    case 64:
        return float64_add(a, b, fs);
    default:
        g_assert_not_reached();
    }
}

static __uint128_t ame_reduction_extreme(__uint128_t a, __uint128_t b,
                                        uint32_t datatype, bool minimum,
                                        float_status *fs)
{
    unsigned size = datatype & 0xff;
    uint8_t a_bits[sizeof(__uint128_t)] = { 0 };
    uint8_t b_bits[sizeof(__uint128_t)] = { 0 };
    int relation;

    ame_store_bits(a_bits, 0, size, a);
    ame_store_bits(b_bits, 0, size, b);
    relation = ame_compare_elements(a_bits, 0, datatype,
                                    b_bits, 0, datatype, fs);
    if (datatype & BIT(8)) {
        uint64_t magnitude = (UINT64_C(1) << (size - 1)) - 1;

        if (relation == 2) {
            return ame_float_default_nan(datatype, fs);
        }
        if (!(a & magnitude) && !(b & magnitude)) {
            /* Mixed-sign zero ties; preserve -0/-0 and +0/+0 as well. */
            return minimum ? a | b : a & b;
        }
    }
    return relation == (minimum ? float_relation_greater :
                                   float_relation_less) ? b : a;
}

void HELPER(ame_reduction)(CPURISCVState *env, uint32_t md, uint32_t ms1,
                           uint32_t selector)
{
    unsigned n = riscv_ame_n(env);
    ame_require_enabled(env, GETPC());
    /* Selectors 0xa3f..0xa48 are ordered col,row pairs in Ztt v0.6. */
    unsigned operation = selector - 0xa3f;
    bool row = operation & 1;
    bool prefix = operation < 4;
    bool add = operation < 2 || (operation >= 4 && operation < 6);
    bool minimum = operation >= 8;
    AMEMOperand d, a;
    uint32_t dtype_d, dtype_a;
    unsigned n_sq;
    uint8_t source[AME_MAX_M_FILE_BYTES];
    uint8_t result[AME_MAX_M_FILE_BYTES];
    float_status fs = env->fp_status;
    bool floating;

    if (md >= riscv_ame_mregs(env) || ms1 >= riscv_ame_mregs(env) ||
        operation >= 10) {
        riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, GETPC());
    }
    dtype_d = env->md[md];
    dtype_a = env->md[ms1];
    floating = ame_reduction_float_type(dtype_d) &&
               ame_reduction_float_type(dtype_a);
    if (!floating && !(ame_integer_datatype_valid(dtype_d) &&
                       ame_integer_datatype_valid(dtype_a))) {
        env->amestatus |= AMESTATUS_UN;
        return;
    }
    ame_m_operand_init(env, &d, dtype_d);
    ame_m_operand_init(env, &a, dtype_a);
    n_sq = MAX(d.pack_factor, a.pack_factor);
    if (!ame_m_operand_form(env, &d, n_sq, md) ||
        !ame_m_operand_form(env, &a, n_sq, ms1)) {
        riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, GETPC());
    }

    memcpy(source, env->mreg, riscv_ame_mfile_bytes(env));
    memcpy(result, env->mreg, riscv_ame_mfile_bytes(env));
    set_float_exception_flags(0, &fs);
    if (floating) {
        ame_float_rounding_from_dtype(&fs, dtype_d);
        set_default_nan_mode(true, &fs);
        set_float_detect_tininess(float_tininess_after_rounding, &fs);
        set_flush_to_zero(false, &fs);
        set_flush_inputs_to_zero(false, &fs);
    }

    /*
     * All possible synchronous exceptions precede arithmetic/status updates.
     * Every square is independent, including packed slots and full wide groups.
     * The FP association tree is a deterministic left fold in ascending index
     * order. The first converted source seeds it (never an implicit +0).
     * Integer conversion and each addition wrap or saturate at the
     * destination width.
     */
    for (unsigned sq = 0; sq < n_sq; sq++) {
        unsigned a_offset, d_offset;
        const uint8_t *src = ame_m_operand_square_buf(env,
            source, ms1, sq, a.size, &a_offset);
        uint8_t *dst = ame_m_operand_square_buf_mut(env,
            result, md, sq, d.size, &d_offset);

        for (unsigned lane = 0; lane < n; lane++) {
            __uint128_t folded = 0;

            for (unsigned k = 0; k < n; k++) {
                unsigned elem = row ? lane * n + k : k * n + lane;
                unsigned offset = a_offset + elem * a.size;
                __uint128_t value = floating ?
                    ame_reduction_float_convert(src, offset, dtype_a,
                                                 dtype_d, &fs) :
                    ame_convert_integer(env, ame_load_bits(src, offset, a.size),
                                        dtype_a, dtype_d);

                if (k == 0) {
                    folded = value;
                } else if (add) {
                    folded = floating ?
                        ame_reduction_float_add(folded, value, dtype_d, &fs) :
                        ame_integer_add(env, folded, value, dtype_d);
                } else {
                    folded = ame_reduction_extreme(folded, value, dtype_d,
                                                   minimum, &fs);
                }
                if (prefix) {
                    ame_store_bits(dst, d_offset + elem * d.size,
                                   d.size, folded);
                }
            }
            if (!prefix) {
                for (unsigned k = 0; k < n; k++) {
                    unsigned elem = row ? lane * n + k :
                                          k * n + lane;

                    ame_store_bits(dst, d_offset + elem * d.size,
                                   d.size, folded);
                }
            }
        }
    }
    memcpy(riscv_ame_mreg(env, md), result + md * riscv_ame_mreg_bytes(env),
           d.span * riscv_ame_mreg_bytes(env));
    ame_commit_float_flags(env, &fs);
}

#include "ame_math_helper.c.inc"
#include "ame_ew_helper.c.inc"

static __uint128_t ame_scatter_value(__uint128_t current,
                                     uint32_t current_dtype,
                                     __uint128_t incoming,
                                     uint32_t incoming_dtype,
                                     uint32_t dst_dtype, bool maximum,
                                     float_status *fs)
{
    uint8_t result[sizeof(__uint128_t)] = { 0 };
    float_status work = *fs;
    float128 cv, iv, value;

    set_float_exception_flags(0, &work);
    set_float_rounding_mode(float_round_to_odd, &work);
    set_default_nan_mode(true, &work);
    set_float_detect_tininess(float_tininess_after_rounding, &work);
    set_flush_to_zero(false, &work);
    set_flush_inputs_to_zero(false, &work);
    ame_store_bits(result, 0, current_dtype & 0xff, current);
    cv = ame_load_float128(result, 0, current_dtype, &work);
    memset(result, 0, sizeof(result));
    ame_store_bits(result, 0, incoming_dtype & 0xff, incoming);
    iv = ame_load_float128(result, 0, incoming_dtype, &work);

    if (maximum) {
        if (float128_is_any_nan(cv) || float128_is_any_nan(iv)) {
            if (float128_is_signaling_nan(cv, &work) ||
                float128_is_signaling_nan(iv, &work)) {
                float_raise(float_flag_invalid, &work);
            }
            value = float128_default_nan(&work);
        } else {
            FloatRelation relation = float128_compare_quiet(cv, iv, &work);

            if (relation == float_relation_less) {
                value = iv;
            } else if (relation == float_relation_equal &&
                       float128_is_zero(cv) && float128_is_zero(iv)) {
                /* max(-0, -0) is -0; a mixed-sign tie selects +0. */
                value = (float128_is_neg(cv) && float128_is_neg(iv)) ?
                        cv : float128_zero;
            } else {
                value = cv;
            }
        }
    } else {
        value = float128_add(cv, iv, &work);
    }
    float_raise(get_float_exception_flags(&work), fs);
    ame_float_rounding_from_dtype(fs, dst_dtype);
    memset(result, 0, sizeof(result));
    ame_store_float128(result, 0, dst_dtype, value, fs);
    return ame_load_bits(result, 0, dst_dtype & 0xff);
}

void HELPER(ame_mrowcol)(CPURISCVState *env, uint32_t md, uint32_t ms1,
                         uint32_t ms2, target_ulong scalar,
                         uint32_t operation)
{
    unsigned n = riscv_ame_n(env);
    ame_require_enabled(env, GETPC());
    bool bcast = operation == AME_MCOLBCAST ||
                 operation == AME_MROWBCAST;
    bool shift = operation == AME_MCOLSHIFT ||
                operation == AME_MROWSHIFT;
    bool gather = operation == AME_MCOLGATHER ||
                 operation == AME_MROWGATHER;
    bool scatter = operation >= AME_MROWSCATADD &&
                   operation <= AME_MCOLSCATMAX;
    bool row = operation == AME_MROWBCAST ||
               operation == AME_MROWGATHER ||
               operation == AME_MROWSHIFT ||
               operation == AME_MROWSCATADD ||
               operation == AME_MROWSCATMAX;
    bool maximum = operation == AME_MROWSCATMAX ||
                   operation == AME_MCOLSCATMAX;
    AMEMOperand d, a = { 0 }, index = { 0 };
    uint8_t source[AME_MAX_M_FILE_BYTES];
    uint8_t result[AME_MAX_M_FILE_BYTES];
    float_status fs = env->fp_status;
    unsigned d_offset, a_offset, index_offset = 0;
    uint8_t *dst;
    const uint8_t *src;
    const uint8_t *indices = NULL;

    if (md >= riscv_ame_mregs(env) || ms2 >= riscv_ame_mregs(env) ||
        ((!bcast && !shift) && ms1 >= riscv_ame_mregs(env)) ||
        (!bcast && !shift && !gather && !scatter)) {
        riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, GETPC());
    }
    if (bcast && (uint32_t)scalar >= n) {
        riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, GETPC());
    }

    if (!ame_rowcol_operand_type(env, md, &d)) {
        return;
    }
    if (bcast || shift) {
        if (!ame_rowcol_operand_type(env, ms2, &a)) {
            return;
        }
        if (env->md[md] != env->md[ms2]) {
            env->amestatus |= AMESTATUS_UN;
            return;
        }
    } else {
        if (!ame_rowcol_operand_type(env, ms1, &a) ||
            !ame_rowcol_operand_type(env, ms2, &index)) {
            return;
        }
        if ((env->md[ms1] ^ env->md[md]) & BIT(8)) {
            /* Cross integer/floating scatter tuples require an explicit
             * conversion contract; do not reinterpret raw bits as FP. */
            env->amestatus |= AMESTATUS_UN;
            return;
        }
        if (!ame_integer_datatype_valid(env->md[ms2]) ||
            (gather && env->md[md] != env->md[ms1])) {
            env->amestatus |= AMESTATUS_UN;
            return;
        }
    }
    ame_rowcol_operand_form(env, md, &d, GETPC());
    if (bcast || shift) {
        ame_rowcol_operand_form(env, ms2, &a, GETPC());
    } else {
        ame_rowcol_operand_form(env, ms1, &a, GETPC());
        ame_rowcol_operand_form(env, ms2, &index, GETPC());
    }

    memcpy(source, env->mreg, riscv_ame_mfile_bytes(env));
    memcpy(result, env->mreg, riscv_ame_mfile_bytes(env));
    dst = ame_m_operand_square_buf_mut(env, result, md, 0, d.size, &d_offset);
    src = ame_m_operand_square_buf(env, source, bcast || shift ? ms2 : ms1,
                                   0, a.size, &a_offset);
    if (gather || scatter) {
        indices = ame_m_operand_square_buf(env, source, ms2, 0, index.size,
                                           &index_offset);
    }
    if (env->md[md] & BIT(8)) {
        ame_float_rounding_from_dtype(&fs, env->md[md]);
    }
    set_float_exception_flags(0, &fs);

    if (bcast || shift || gather) {
        int32_t amount = (int32_t)(uint32_t)scalar;

        for (unsigned i = 0; i < n; i++) {
            for (unsigned j = 0; j < n; j++) {
                unsigned source_row = i;
                unsigned source_col = j;
                bool in_range = true;

                if (bcast) {
                    if (row) {
                        source_row = (uint32_t)scalar;
                    } else {
                        source_col = (uint32_t)scalar;
                    }
                } else if (shift) {
                    int64_t coordinate = (int64_t)(row ? i : j) + amount;

                    in_range = coordinate >= 0 && coordinate < n;
                    if (row) {
                        source_row = coordinate;
                    } else {
                        source_col = coordinate;
                    }
                } else {
                    unsigned elem = i * n + j;
                    unsigned selected = (uint32_t)ame_load_bits(
                        indices, index_offset + elem * index.size,
                        MIN(index.size, 32)) & (n - 1);

                    if (row) {
                        source_row = selected;
                    } else {
                        source_col = selected;
                    }
                }

                __uint128_t value = in_range ? ame_load_bits(
                    src, a_offset + (source_row * n + source_col) *
                         a.size, a.size) : 0;
                ame_store_bits(dst, d_offset + (i * n + j) * d.size,
                               d.size, value);
            }
        }
    } else {
        for (unsigned elem = 0; elem < riscv_ame_nelem(env); elem++) {
            unsigned i = elem / n;
            unsigned j = elem % n;
            unsigned selected = (uint32_t)ame_load_bits(
                indices, index_offset + elem * index.size,
                MIN(index.size, 32)) & (n - 1);
            unsigned target = row ? selected * n + j :
                                    i * n + selected;
            __uint128_t incoming = ame_load_bits(
                src, a_offset + elem * a.size, a.size);
            __uint128_t current = ame_load_bits(
                dst, d_offset + target * d.size, d.size);

            if (!(env->md[ms1] & BIT(8)) && !(env->md[md] & BIT(8))) {
                AMEExactInteger cv = ame_exact_integer(current, env->md[md]);
                AMEExactInteger iv = ame_exact_integer(incoming, env->md[ms1]);
                AMEExactInteger value;
                bool saturated = false;

                if (maximum) {
                    int cmp = ame_exact_magnitude_compare(cv, iv);
                    bool take = cv.negative != iv.negative ? cv.negative :
                                cv.negative ? cmp > 0 : cmp < 0;

                    value = take ? iv : cv;
                } else {
                    value = ame_exact_add(cv, iv);
                }
                ame_store_bits(dst, d_offset + target * d.size, d.size,
                               ame_exact_convert(value, env->md[md],
                                                 &saturated));
                if (saturated) {
                    env->amexsat |= AMEXSAT_SAT;
                }
                continue;
            }
            ame_store_bits(dst, d_offset + target * d.size, d.size,
                           ame_scatter_value(current, env->md[md],
                                             incoming,
                                             env->md[ms1], env->md[md],
                                             maximum, &fs));
        }
    }

    memcpy(riscv_ame_mreg(env, md), &result[md * riscv_ame_mreg_bytes(env)],
           d.span * riscv_ame_mreg_bytes(env));
    ame_commit_float_flags(env, &fs);
}

static void ame_copy_square_element(uint8_t *dst, unsigned dst_offset,
                                    unsigned dst_elem, const uint8_t *src,
                                    unsigned src_offset, unsigned src_elem,
                                    unsigned size)
{
    ame_store_bits(dst, dst_offset + dst_elem * size, size,
                   ame_load_bits(src, src_offset + src_elem * size, size));
}

void HELPER(ame_mrowcol_zip)(CPURISCVState *env, uint32_t ms1,
                             uint32_t ms2, uint32_t operation)
{
    unsigned n = riscv_ame_n(env);
    ame_require_enabled(env, GETPC());
    bool unzip = operation == AME_MCOLUNZIP ||
                 operation == AME_MROWUNZIP;
    bool row = operation == AME_MROWZIP || operation == AME_MROWUNZIP;
    AMEMOperand first, second;
    uint8_t source[AME_MAX_M_FILE_BYTES];
    uint8_t result[AME_MAX_M_FILE_BYTES];
    unsigned a_offset, b_offset, x_offset, y_offset;
    const uint8_t *a;
    const uint8_t *b;
    uint8_t *x;
    uint8_t *y;

    if (ms1 >= riscv_ame_mregs(env) || ms2 >= riscv_ame_mregs(env) ||
        ms1 == ms2 ||
        (operation != AME_MCOLUNZIP && operation != AME_MCOLZIP &&
         operation != AME_MROWUNZIP && operation != AME_MROWZIP)) {
        riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, GETPC());
    }
    if (!ame_rowcol_operand_type(env, ms1, &first) ||
        !ame_rowcol_operand_type(env, ms2, &second)) {
        return;
    }
    if (env->md[ms1] != env->md[ms2]) {
        env->amestatus |= AMESTATUS_UN;
        return;
    }
    ame_rowcol_operand_form(env, ms1, &first, GETPC());
    ame_rowcol_operand_form(env, ms2, &second, GETPC());

    memcpy(source, env->mreg, riscv_ame_mfile_bytes(env));
    memcpy(result, env->mreg, riscv_ame_mfile_bytes(env));
    a = ame_m_operand_square_buf(env, source, ms1, 0, first.size, &a_offset);
    b = ame_m_operand_square_buf(env, source, ms2, 0, second.size, &b_offset);
    x = ame_m_operand_square_buf_mut(env, result, ms1, 0, first.size,
                                     &x_offset);
    y = ame_m_operand_square_buf_mut(env, result, ms2, 0, second.size,
                                     &y_offset);

    for (unsigned i = 0; i < n; i++) {
        for (unsigned j = 0; j < n / 2; j++) {
            unsigned lo = row ? j * n + i : i * n + j;
            unsigned hi = row ? (n / 2 + j) * n + i :
                                i * n + n / 2 + j;
            unsigned even = row ? (2 * j) * n + i :
                                  i * n + 2 * j;
            unsigned odd = even + (row ? n : 1);

            if (unzip) {
                ame_copy_square_element(x, x_offset, lo, a, a_offset,
                                        even, first.size);
                ame_copy_square_element(x, x_offset, hi, b, b_offset,
                                        even, first.size);
                ame_copy_square_element(y, y_offset, lo, a, a_offset,
                                        odd, first.size);
                ame_copy_square_element(y, y_offset, hi, b, b_offset,
                                        odd, first.size);
            } else {
                ame_copy_square_element(x, x_offset, even, a, a_offset,
                                        lo, first.size);
                ame_copy_square_element(x, x_offset, odd, b, b_offset,
                                        lo, first.size);
                ame_copy_square_element(y, y_offset, even, a, a_offset,
                                        hi, first.size);
                ame_copy_square_element(y, y_offset, odd, b, b_offset,
                                        hi, first.size);
            }
        }
    }

    memcpy(riscv_ame_mreg(env, ms1), &result[ms1 * riscv_ame_mreg_bytes(env)],
           first.span * riscv_ame_mreg_bytes(env));
    memcpy(riscv_ame_mreg(env, ms2), &result[ms2 * riscv_ame_mreg_bytes(env)],
           second.span * riscv_ame_mreg_bytes(env));
}
