
/*
 * Ztt special mathematical operations; included by matrix_helper.c.
 * No host floating-point arithmetic is used in the execution path.
 */

enum {
    AME_MATH_LDEXP = 0x3f,
    AME_MATH_LDEXP_X = 0x40,
    AME_MATH_LDEXPACC = 0x41,
    AME_MATH_LDEXPACC_X = 0x42,
    AME_MATH_RDEXP = 0x45,
    AME_MATH_RDEXPACC = 0x46,
    AME_MATH_REC = 0xa31,
    AME_MATH_RSQRT = 0xa32,
    AME_MATH_SQRT = 0xa34,
    AME_MATH_COS = 0xa2e,
    AME_MATH_EXP2 = 0xa2f,
    AME_MATH_LOG2 = 0xa30,
    AME_MATH_SIN = 0xa33,
    AME_MATH_TANH = 0xa35,
    AME_MATH_FRINTM = 0xa25,
    AME_MATH_FRINTN = 0xa26,
    AME_MATH_FRINTP = 0xa27,
    AME_MATH_FRINTZ = 0xa28,
    AME_MATH_LOG2SUB = 0x43,
    AME_MATH_LOG2SUB_X = 0x44,
    AME_MATH_SUBLOG2 = 0x47,
    AME_MATH_SUBLOG2_X = 0x48,
};

static float128 ame_math_load(const uint8_t *src, unsigned offset,
                              uint32_t dtype, float_status *fs)
{
    return float64_to_float128(ame_load_float64(src, offset, dtype, fs), fs);
}

static uint64_t ame_math_round(float128 value, uint32_t dtype,
                               float_status *fs)
{
    unsigned size = dtype & 0xff;

    if (size == 64) {
        return float128_to_float64(value, fs);
    }
    if (size == 32) {
        return float128_to_float32(value, fs);
    }
    /* Round-to-odd staging avoids double rounding on the final narrowing. */
    float_status stage = *fs;
    float64 intermediate;

    set_float_rounding_mode(float_round_to_odd, &stage);
    set_float_exception_flags(0, &stage);
    intermediate = float128_to_float64(value, &stage);
    return ((dtype >> 26) & 0x1f) == 8 ?
           float64_to_bfloat16(intermediate, fs) :
           float64_to_float16(intermediate, true, fs);
}

static int ame_math_exponent(__uint128_t value, uint32_t dtype, bool negate)
{
    unsigned size = dtype & 0xff;
    bool negative = (dtype & BIT(30)) &&
                    (value & ((__uint128_t)1 << (size - 1)));
    __uint128_t mask = size == 128 ? ~(__uint128_t)0 :
                      (((__uint128_t)1 << size) - 1);
    __uint128_t magnitude = negative ? (-value & mask) : value;
    int exponent = magnitude > 4096 ? 4096 : (int)magnitude;

    /* Even the largest binary64 cancellation cannot matter past this bound. */
    return negative != negate ? -exponent : exponent;
}

/*
 * Deterministic FP16/FP32 transcendental kernels. Series are evaluated at
 * binary128 precision, not with host libm. Trigonometric range reduction uses
 * a 256-fraction-bit modulus so even the largest finite FP32 input is covered.
 */
static float128 ame_math_int(int value, float_status *fs)
{
    return float64_to_float128(int32_to_float64(value, fs), fs);
}

static const float128 ame_math_ln2 = {
    .high = UINT64_C(0x3ffe62e42fefa39e), .low = UINT64_C(0xf35793c7673007e6)
};
static const float128 ame_math_pi = {
    .high = UINT64_C(0x4000921fb54442d1), .low = UINT64_C(0x8469898cc51701b8)
};

/*
 * Move one binary128 ulp in magnitude, retaining the side of a tiny correction
 * which would otherwise disappear before directed FP16/FP32 rounding.
 */
static float128 ame_math_adjacent(float128 x, bool increase)
{
    if (increase) {
        x.low++;
        x.high += x.low == 0;
    } else {
        x.high -= x.low == 0;
        x.low--;
    }
    return x;
}

static bool ame_math_tiny(float128 x)
{
    return ((x.high >> 48) & 0x7fff) < 16383 - 32;
}

#define AME_MATH_REDUCE_WORDS 5

static bool ame_math_words_ge(const uint64_t *a, const uint64_t *b)
{
    for (int i = AME_MATH_REDUCE_WORDS - 1; i >= 0; i--) {
        if (a[i] != b[i]) {
            return a[i] > b[i];
        }
    }
    return true;
}

static void ame_math_words_sub(uint64_t *dest, const uint64_t *a,
                                const uint64_t *b)
{
    bool borrow = false;

    for (unsigned i = 0; i < AME_MATH_REDUCE_WORDS; i++) {
        uint64_t sub = b[i] + borrow;
        bool carry = sub < b[i];
        uint64_t av = a[i];

        dest[i] = av - sub;
        borrow = carry || av < sub;
    }
}

static void ame_math_words_shift(uint64_t *words, unsigned bit)
{
    for (unsigned i = 0; i < AME_MATH_REDUCE_WORDS; i++) {
        unsigned carry = words[i] >> 63;

        words[i] = (words[i] << 1) | bit;
        bit = carry;
    }
}

static float128 ame_math_reduce_angle(float128 x, float_status *fs)
{
    /*
     * floor(2*pi*2^256), least significant word first. The constant's error
     * contributes less than 2^-130 radians over the finite FP32 input range.
     */
    static const uint64_t two_pi[AME_MATH_REDUCE_WORDS] = {
        UINT64_C(0x105df531d89cd912), UINT64_C(0x48127044533e63a0),
        UINT64_C(0x2633145c06e0e689), UINT64_C(0x487ed5110b4611a6), 6,
    };
    uint64_t remainder[AME_MATH_REDUCE_WORDS] = { 0 };
    uint64_t twice[AME_MATH_REDUCE_WORDS];
    bool negative = float128_is_neg(x);
    float128 magnitude = float128_abs(x);
    float128 reduced = float128_zero;
    int exponent = ((x.high >> 48) & 0x7fff) - 16383;
    int shift = 256 + exponent - 112;
    __uint128_t significand;

    if (float128_compare_quiet(magnitude, ame_math_pi, fs) <= 0) {
        return x;
    }
    significand = ((__uint128_t)((x.high & UINT64_C(0x0000ffffffffffff)) |
                                  UINT64_C(0x0001000000000000)) << 64) | x.low;
    /* Exact binary long division of abs(x)*2^256 by the fixed-point modulus. */
    for (int bit = exponent + 256; bit >= 0; bit--) {
        unsigned input = bit >= shift ? (significand >> (bit - shift)) & 1 : 0;

        ame_math_words_shift(remainder, input);
        if (ame_math_words_ge(remainder, two_pi)) {
            ame_math_words_sub(remainder, remainder, two_pi);
        }
    }
    memcpy(twice, remainder, sizeof(twice));
    ame_math_words_shift(twice, 0);
    if (ame_math_words_ge(twice, two_pi)) {
        ame_math_words_sub(remainder, two_pi, remainder);
        negative = !negative;
    }
    for (int i = AME_MATH_REDUCE_WORDS - 1; i >= 0; i--) {
        reduced = float128_add(float128_scalbn(reduced, 64, fs),
                               uint64_to_float128(remainder[i], fs), fs);
    }
    reduced = float128_scalbn(reduced, -256, fs);
    return negative ? float128_chs(reduced) : reduced;
}

static float128 ame_math_exp2(float128 x, float_status *fs)
{
    float128 one = ame_math_int(1, fs);
    float128 whole, fraction, term, sum;
    int n;

    if (float128_is_any_nan(x)) {
        return float128_add(x, x, fs);
    }
    if (float128_is_infinity(x)) {
        return float128_is_neg(x) ? float128_zero : x;
    }
    if (!float128_is_zero(x) && ame_math_tiny(x)) {
        float_raise(float_flag_inexact, fs);
        return ame_math_adjacent(one, !float128_is_neg(x));
    }
    if (float128_compare(x, ame_math_int(4096, fs), fs) > 0) {
        x = ame_math_int(4096, fs);
    } else if (float128_compare(x, ame_math_int(-4096, fs), fs) < 0) {
        x = ame_math_int(-4096, fs);
    }
    /* nearest integer gives |fraction * ln(2)| <= ln(2)/2 */
    set_float_rounding_mode(float_round_nearest_even, fs);
    whole = float128_round_to_int(x, fs);
    n = float128_to_int32(whole, fs);
    fraction = float128_sub(x, whole, fs);
    if (float128_is_zero(fraction)) {
        return float128_scalbn(one, n, fs);
    }
    fraction = float128_mul(fraction, ame_math_ln2, fs);
    term = sum = one;
    for (int k = 1; k <= 80; k++) {
        term = float128_div(float128_mul(term, fraction, fs),
                            ame_math_int(k, fs), fs);
        sum = float128_add(sum, term, fs);
    }
    return float128_scalbn(sum, n, fs);
}

static float128 ame_math_log2(float128 x, float_status *fs)
{
    float128 one = ame_math_int(1, fs);
    float128 z, z2, term, sum;
    int exponent;

    if (float128_is_any_nan(x) || float128_is_zero(x) ||
        float128_is_neg(x) || float128_is_infinity(x)) {
        /* softfloat log2 handles precisely these special values. */
        return float64_to_float128(float64_log2(float128_to_float64(x, fs),
                                                fs), fs);
    }
    exponent = ((x.high >> 48) & 0x7fff) - 16383;
    x.high = (x.high & UINT64_C(0x0000ffffffffffff)) |
             UINT64_C(0x3fff000000000000);
    if (float128_compare_quiet(x, one, fs) == float_relation_equal) {
        return ame_math_int(exponent, fs);
    }
    /* log(m) = 2 * atanh((m-1)/(m+1)); 1 <= m < 2, hence |z| < 1/3. */
    z = float128_div(float128_sub(x, one, fs), float128_add(x, one, fs), fs);
    z2 = float128_mul(z, z, fs);
    term = sum = z;
    for (int k = 3; k <= 201; k += 2) {
        term = float128_mul(term, z2, fs);
        sum = float128_add(sum, float128_div(term, ame_math_int(k, fs), fs), fs);
    }
    sum = float128_div(float128_scalbn(sum, 1, fs), ame_math_ln2, fs);
    return float128_add(ame_math_int(exponent, fs), sum, fs);
}

static float128 ame_math_sincos(float128 x, bool cosine, float_status *fs)
{
    float128 one = ame_math_int(1, fs);
    float128 x2, term, sum;

    if (float128_is_any_nan(x)) {
        return float128_add(x, x, fs);
    }
    if (float128_is_infinity(x)) {
        float_raise(float_flag_invalid, fs);
        return float128_default_nan(fs);
    }
    if (float128_is_zero(x)) {
        return cosine ? one : x;
    }
    if (ame_math_tiny(x)) {
        float_raise(float_flag_inexact, fs);
        return ame_math_adjacent(cosine ? one : x, false);
    }
    set_float_rounding_mode(float_round_nearest_even, fs);
    x = ame_math_reduce_angle(x, fs);
    x2 = float128_chs(float128_mul(x, x, fs));
    term = sum = cosine ? one : x;
    for (int k = 1; k <= 80; k++) {
        int divisor = cosine ? (2 * k - 1) * (2 * k) : (2 * k) * (2 * k + 1);

        term = float128_div(float128_mul(term, x2, fs),
                            ame_math_int(divisor, fs), fs);
        sum = float128_add(sum, term, fs);
    }
    float_raise(float_flag_inexact, fs);
    return sum;
}

static float128 ame_math_tanh(float128 x, float_status *fs)
{
    float128 one = ame_math_int(1, fs);
    bool negative = float128_is_neg(x);
    float128 y;

    if (float128_is_any_nan(x)) {
        return float128_add(x, x, fs);
    }
    if (float128_is_zero(x)) {
        return x;
    }
    if (float128_is_infinity(x)) {
        return negative ? float128_chs(one) : one;
    }
    if (ame_math_tiny(x)) {
        float_raise(float_flag_inexact, fs);
        return ame_math_adjacent(x, false);
    }
    x = float128_abs(x);
    if (float128_compare_quiet(x, ame_math_int(32, fs), fs) > 0) {
        /* Strictly below one; preserve directed rounding at target precision. */
        y = (float128) { .high = UINT64_C(0x3ffeffffffffffff),
                         .low = UINT64_MAX };
    } else {
        y = ame_math_exp2(float128_div(float128_scalbn(x, 1, fs),
                                       ame_math_ln2, fs), fs);
        y = float128_div(float128_sub(y, one, fs), float128_add(y, one, fs), fs);
    }
    float_raise(float_flag_inexact, fs);
    return negative ? float128_chs(y) : y;
}

static bool ame_math_transcendental_type(uint32_t dtype)
{
    return ame_reduction_float_type(dtype) &&
           (((dtype & 0xff) == 16 && ((dtype >> 26) & 31) == 5) ||
            ((dtype & 0xff) == 32 && ((dtype >> 26) & 31) == 8));
}

void HELPER(ame_math)(CPURISCVState *env, uint32_t md, uint32_t ms1,
                      uint32_t ms2, target_ulong scalar, uint32_t op)
{
    ame_require_enabled(env, GETPC());
    bool unary = op >= 0xa00;
    bool scalar_form = op == AME_MATH_LDEXP_X || op == AME_MATH_LDEXPACC_X ||
                       op == AME_MATH_LOG2SUB_X || op == AME_MATH_SUBLOG2_X;
    bool accumulate = op == AME_MATH_LDEXPACC ||
                      op == AME_MATH_LDEXPACC_X || op == AME_MATH_RDEXPACC;
    bool negate = op == AME_MATH_RDEXP || op == AME_MATH_RDEXPACC;
    bool log_bias = op == AME_MATH_LOG2SUB || op == AME_MATH_LOG2SUB_X ||
                    op == AME_MATH_SUBLOG2 || op == AME_MATH_SUBLOG2_X;
    bool transcendental = log_bias || op == AME_MATH_COS ||
                          op == AME_MATH_SIN || op == AME_MATH_EXP2 ||
                          op == AME_MATH_LOG2 || op == AME_MATH_TANH;
    unsigned data_reg = scalar_form ? ms2 : ms1;
    uint32_t dtype_d, dtype_a, dtype_b = AME_MAX_INT_DTYPE;
    AMEMOperand d, a, b = { 0 };
    uint8_t source[AME_MAX_M_FILE_BYTES];
    uint8_t result[AME_MAX_M_FILE_BYTES];
    float_status fs = env->fp_status;
    unsigned n_sq;

    if (md >= riscv_ame_mregs(env) || data_reg >= riscv_ame_mregs(env) ||
        (!unary && !scalar_form && ms2 >= riscv_ame_mregs(env))) {
        riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, GETPC());
    }
    dtype_d = env->md[md];
    dtype_a = env->md[data_reg];
    if (!unary && !scalar_form) {
        dtype_b = env->md[ms2];
    } else if (log_bias) {
        dtype_b = env->amestype;
    }
    /* Accepted data tuples are standard IEEE formats with Inf=Denorm=1. */
    if (!ame_reduction_float_type(dtype_d) ||
        !ame_reduction_float_type(dtype_a) ||
        (!unary && !log_bias && !ame_integer_datatype_valid(dtype_b)) ||
        (log_bias && !ame_reduction_float_type(dtype_b)) ||
        (transcendental &&
         (!ame_math_transcendental_type(dtype_d) ||
          !ame_math_transcendental_type(dtype_a) ||
          (log_bias && !ame_math_transcendental_type(dtype_b))))) {
        env->amestatus |= AMESTATUS_UN;
        return;
    }
    ame_m_operand_init(env, &d, dtype_d);
    ame_m_operand_init(env, &a, dtype_a);
    n_sq = MAX(d.pack_factor, a.pack_factor);
    if (!unary && !scalar_form) {
        ame_m_operand_init(env, &b, dtype_b);
        n_sq = MAX(n_sq, b.pack_factor);
    }
    if (!ame_m_operand_form(env, &d, n_sq, md) ||
        !ame_m_operand_form(env, &a, n_sq, data_reg) ||
        (!unary && !scalar_form && !ame_m_operand_form(env, &b, n_sq, ms2))) {
        riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, GETPC());
    }
    memcpy(source, env->mreg, riscv_ame_mfile_bytes(env));
    memcpy(result, env->mreg, riscv_ame_mfile_bytes(env));
    ame_float_rounding_from_dtype(&fs, dtype_d);
    set_float_exception_flags(0, &fs);
    set_default_nan_mode(true, &fs);
    set_float_detect_tininess(float_tininess_after_rounding, &fs);
    set_flush_to_zero(false, &fs);
    set_flush_inputs_to_zero(false, &fs);

    for (unsigned sq = 0; sq < n_sq; sq++) {
        unsigned ao, bo = 0, dest_offset;
        const uint8_t *ap = ame_m_operand_square_buf(env, source, data_reg, sq,
                                                    a.size, &ao);
        const uint8_t *bp = (!unary && !scalar_form) ?
            ame_m_operand_square_buf(env, source, ms2, sq, b.size, &bo) : NULL;
        uint8_t *dp = ame_m_operand_square_buf_mut(env, result, md, sq, d.size,
                                                  &dest_offset);

        for (unsigned i = 0; i < riscv_ame_nelem(env); i++) {
            float_status work = fs;
            float128 av, value;
            float128 one = { .high = UINT64_C(0x3fff000000000000), .low = 0 };
            unsigned offset = dest_offset + i * d.size;

            set_float_exception_flags(0, &work);
            set_float_rounding_mode(float_round_to_odd, &work);
            av = ame_math_load(ap, ao + i * a.size, dtype_a, &work);
            if (unary) {
                switch (op) {
                case AME_MATH_COS:
                    value = ame_math_sincos(av, true, &work);
                    break;
                case AME_MATH_EXP2:
                    value = ame_math_exp2(av, &work);
                    break;
                case AME_MATH_LOG2:
                    value = ame_math_log2(av, &work);
                    break;
                case AME_MATH_REC:
                    value = float128_div(one, av, &work);
                    break;
                case AME_MATH_SQRT:
                    value = float128_sqrt(av, &work);
                    break;
                case AME_MATH_RSQRT:
                    value = float128_div(one, float128_sqrt(av, &work), &work);
                    break;
                case AME_MATH_SIN:
                    value = ame_math_sincos(av, false, &work);
                    break;
                case AME_MATH_TANH:
                    value = ame_math_tanh(av, &work);
                    break;
                case AME_MATH_FRINTM:
                case AME_MATH_FRINTN:
                case AME_MATH_FRINTP:
                case AME_MATH_FRINTZ: {
                    static const FloatRoundMode modes[] = {
                        float_round_down, float_round_nearest_even,
                        float_round_up, float_round_to_zero,
                    };
                    FloatRoundMode mode = modes[op - AME_MATH_FRINTM];

                    set_float_rounding_mode(mode, &work);
                    set_float_rounding_mode(mode, &fs);
                    /* Preserve NX when rounding discards fractional bits. */
                    value = float128_round_to_int(av, &work);
                    break;
                }
                default:
                    g_assert_not_reached();
                }
            } else if (log_bias) {
                uint8_t scalar_bits[16] = { 0 };
                float128 bias, logarithm;

                if (scalar_form) {
                    __uint128_t bits = ame_convert_scalar(env, scalar,
                                                          dtype_b, dtype_a,
                                                          &work);

                    ame_store_bits(scalar_bits, 0, dtype_a & 255, bits);
                    bias = ame_math_load(scalar_bits, 0, dtype_a, &work);
                    set_float_rounding_mode(float_round_to_odd, &work);
                } else {
                    bias = ame_math_load(bp, bo + i * b.size, dtype_b, &work);
                }
                logarithm = ame_math_log2(float128_abs(av), &work);
                set_float_rounding_mode(get_float_rounding_mode(&fs), &work);
                value = (op == AME_MATH_SUBLOG2 || op == AME_MATH_SUBLOG2_X) ?
                        float128_sub(bias, logarithm, &work) :
                        float128_sub(logarithm, bias, &work);
            } else {
                __uint128_t bits = scalar_form ? (__uint128_t)scalar :
                    ame_load_bits(bp, bo + i * b.size, b.size);
                int exponent = ame_math_exponent(bits, dtype_b, negate);

                /* Scaling is exact in binary128 for all finite IEEE inputs. */
                value = float128_scalbn(av, exponent, &work);
                if (accumulate) {
                    float128 seed = ame_math_load(dp, offset, dtype_d, &work);

                    if (float128_is_zero(seed) && float128_is_zero(value)) {
                        set_float_rounding_mode(get_float_rounding_mode(&fs),
                                                &work);
                    }
                    /* Round-to-odd retains the sticky bit for final rounding. */
                    value = float128_add(seed, value, &work);
                }
            }
            uint64_t bits = ame_math_round(value, dtype_d, &fs);

            /* Exact cancellation zero must follow the destination rounding. */
            if (accumulate && float128_is_zero(value) &&
                !float128_is_zero(av)) {
                bits = get_float_rounding_mode(&fs) == float_round_down ?
                       UINT64_C(1) << (d.size - 1) : 0;
            }
            float_raise(get_float_exception_flags(&work), &fs);
            ame_store_bits(dp, offset, d.size, bits);
        }
    }
    memcpy(riscv_ame_mreg(env, md), result + md * riscv_ame_mreg_bytes(env),
           d.span * riscv_ame_mreg_bytes(env));
    ame_commit_float_flags(env, &fs);
}
