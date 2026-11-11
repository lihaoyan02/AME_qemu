
/*
 * Ztt elementwise arithmetic operations; included by matrix_helper.c after
 * the shared exact-integer and SoftFloat conversion helpers.
 */

static AMEExactInteger ame_exact_negate(AMEExactInteger value)
{
    if (value.lo || value.hi) {
        value.negative = !value.negative;
    }
    return value;
}

static int ame_exact_compare(AMEExactInteger a, AMEExactInteger b)
{
    int magnitude;

    if (a.negative != b.negative) {
        return a.negative ? -1 : 1;
    }
    magnitude = ame_exact_magnitude_compare(a, b);
    return a.negative ? -magnitude : magnitude;
}

static void ame_exact_increment(AMEExactInteger *value)
{
    value->lo++;
    if (!value->lo) {
        value->hi++;
    }
}

/* Round an exact value divided by two using the destination integer mode. */
static AMEExactInteger ame_exact_half(AMEExactInteger value, uint32_t dtype)
{
    bool discarded = value.lo & 1;
    unsigned mode = (dtype >> 27) & 3;
    AMEExactInteger result = {
        .lo = (value.lo >> 1) | (value.hi << 127),
        .hi = value.hi >> 1,
        .negative = value.negative,
    };
    bool increment = false;

    if (discarded) {
        switch (mode) {
        case 0: /* RNU: nearest, ties toward +infinity. */
            increment = !value.negative;
            break;
        case 1: /* RNE: select the even neighbor. */
            increment = result.lo & 1;
            break;
        case 2: /* RDN: arithmetic floor. */
            increment = value.negative;
            break;
        case 3: /* ROD: jam any discarded fraction into the result LSB. */
            increment = !(result.lo & 1);
            break;
        }
    }
    if (increment) {
        ame_exact_increment(&result);
    }
    result.negative &= result.lo != 0 || result.hi != 0;
    return result;
}

static bool ame_ew_operation_valid(uint32_t op)
{
    return op == AME_MABS || (op <= AME_MSUB + 1);
}

static bool ame_ew_reads_dest(uint32_t op)
{
    op &= ~1;
    return op == AME_MMULACC_EW || op == AME_MMULACCNEG_EW ||
           op == AME_MMULADD || op == AME_MMULSUB;
}

static AMEExactInteger ame_ew_integer_value(uint32_t op,
                                            AMEExactInteger a,
                                            AMEExactInteger b,
                                            AMEExactInteger d,
                                            uint32_t dtype_d)
{
    AMEExactInteger product;

    switch (op & ~1) {
    case AME_MABSDIFF:
        a = ame_exact_add(a, ame_exact_negate(b));
        a.negative = false;
        return a;
    case AME_MADD:
        return ame_exact_add(a, b);
    case AME_MHDIFF:
        return ame_exact_half(ame_exact_add(b, ame_exact_negate(a)),
                              dtype_d);
    case AME_MMAX:
        return ame_exact_compare(a, b) < 0 ? b : a;
    case AME_MMEAN:
        return ame_exact_half(ame_exact_add(a, b), dtype_d);
    case AME_MMIN:
        return ame_exact_compare(a, b) > 0 ? b : a;
    case AME_MMUL:
        return ame_exact_product(a, b);
    case AME_MMULACC_EW:
        return ame_exact_add(d, ame_exact_product(a, b));
    case AME_MMULACCNEG_EW:
        product = ame_exact_negate(ame_exact_product(a, b));
        return ame_exact_add(d, product);
    case AME_MMULADD:
        return ame_exact_add(a, ame_exact_product(b, d));
    case AME_MMULNEG:
        return ame_exact_negate(ame_exact_product(a, b));
    case AME_MMULSUB:
        product = ame_exact_negate(ame_exact_product(b, d));
        return ame_exact_add(a, product);
    case AME_MSUB:
        return ame_exact_add(b, ame_exact_negate(a));
    default:
        g_assert_not_reached();
    }
}

static float128 ame_ew_float_value(uint32_t op, float128 a, float128 b,
                                   float128 d, float_status *fs)
{
    float128 value;

    switch (op & ~1) {
    case AME_MABSDIFF:
        return float128_abs(float128_sub(a, b, fs));
    case AME_MADD:
        return float128_add(a, b, fs);
    case AME_MHDIFF:
        return float128_scalbn(float128_sub(b, a, fs), -1, fs);
    case AME_MMAX:
        return float128_max(a, b, fs);
    case AME_MMEAN:
        return float128_scalbn(float128_add(a, b, fs), -1, fs);
    case AME_MMIN:
        return float128_min(a, b, fs);
    case AME_MMUL:
        return float128_mul(a, b, fs);
    case AME_MMULACC_EW:
        return float128_muladd(a, b, d, 0, fs);
    case AME_MMULACCNEG_EW:
        return float128_muladd(a, b, d, float_muladd_negate_product, fs);
    case AME_MMULADD:
        return float128_muladd(b, d, a, 0, fs);
    case AME_MMULNEG:
        value = float128_mul(a, b, fs);
        return float128_is_any_nan(value) ? value : float128_chs(value);
    case AME_MMULSUB:
        return float128_muladd(b, d, a, float_muladd_negate_product, fs);
    case AME_MSUB:
        return float128_sub(b, a, fs);
    default:
        g_assert_not_reached();
    }
}

void HELPER(ame_ew_arith)(CPURISCVState *env, uint32_t md, uint32_t ms1,
                          uint32_t ms2, target_ulong scalar, uint32_t op)
{
    ame_require_enabled(env, GETPC());
    bool unary = op == AME_MABS;
    bool scalar_form = !unary && (op & 1);
    uint32_t dtype_d, dtype_a, dtype_b = 0;
    bool integer, floating;
    AMEMOperand d, a = { 0 }, b = { 0 };
    unsigned n_sq;
    uint8_t source[AME_MAX_M_FILE_BYTES];
    uint8_t result[AME_MAX_M_FILE_BYTES];
    uint8_t scalar_value[sizeof(__uint128_t)] = { 0 };
    bool saturated = false;
    float_status fs = env->fp_status;

    if (md >= riscv_ame_mregs(env) || ms1 >= riscv_ame_mregs(env) ||
        ms2 >= riscv_ame_mregs(env) || !ame_ew_operation_valid(op)) {
        riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, GETPC());
    }

    dtype_d = env->md[md];
    dtype_a = scalar_form ? env->amestype : env->md[ms1];
    if (!unary) {
        dtype_b = env->md[ms2];
    }
    integer = ame_integer_datatype_valid(dtype_d) &&
              ame_integer_datatype_valid(dtype_a) &&
              (unary || ame_integer_datatype_valid(dtype_b));
    floating = ame_reduction_float_type(dtype_d) &&
               ame_reduction_float_type(dtype_a) &&
               (unary || ame_reduction_float_type(dtype_b));
    if ((!integer && !floating) ||
        (((!unary && (op & ~1) == AME_MMAX) ||
          (!unary && (op & ~1) == AME_MMIN)) &&
         (dtype_d != dtype_b || (!scalar_form && dtype_a != dtype_b)))) {
        env->amestatus |= AMESTATUS_UN;
        return;
    }

    ame_m_operand_init(env, &d, dtype_d);
    if (!scalar_form) {
        ame_m_operand_init(env, &a, dtype_a);
    }
    if (!unary) {
        ame_m_operand_init(env, &b, dtype_b);
    }
    n_sq = d.pack_factor;
    if (!scalar_form) {
        n_sq = MAX(n_sq, a.pack_factor);
    }
    if (!unary) {
        n_sq = MAX(n_sq, b.pack_factor);
    }
    if (!ame_m_operand_form(env, &d, n_sq, md) ||
        (!scalar_form && !ame_m_operand_form(env, &a, n_sq, ms1)) ||
        (!unary && !ame_m_operand_form(env, &b, n_sq, ms2))) {
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

    if (scalar_form) {
        unsigned scalar_size = dtype_a & 0xff;
        __uint128_t scalar_bits = scalar;

        if (scalar_size < 128) {
            scalar_bits &= (((__uint128_t)1 << scalar_size) - 1);
        }
        if (integer) {
            __uint128_t converted = ame_exact_convert(
                ame_exact_integer(scalar_bits, dtype_a), dtype_b, &saturated);

            ame_store_bits(scalar_value, 0, b.size, converted);
        } else {
            uint8_t ingress[sizeof(__uint128_t)] = { 0 };
            float_status convert = fs;
            float128 value;

            ame_store_bits(ingress, 0, scalar_size, scalar_bits);
            ame_float_rounding_from_dtype(&convert, dtype_b);
            set_float_exception_flags(0, &convert);
            value = ame_math_load(ingress, 0, dtype_a, &convert);
            ame_store_bits(scalar_value, 0, b.size,
                           ame_math_round(value, dtype_b, &convert));
            float_raise(get_float_exception_flags(&convert), &fs);
        }
    }

    for (unsigned sq = 0; sq < n_sq; sq++) {
        unsigned ao = 0, bo = 0, old_do, new_do;
        const uint8_t *ap = scalar_form ? scalar_value :
            ame_m_operand_square_buf(env, source, ms1, sq, a.size, &ao);
        const uint8_t *bp = unary ? NULL :
            ame_m_operand_square_buf(env, source, ms2, sq, b.size, &bo);
        const uint8_t *old_dp = ame_m_operand_square_buf(env,
            source, md, sq, d.size, &old_do);
        uint8_t *new_dp = ame_m_operand_square_buf_mut(env,
            result, md, sq, d.size, &new_do);

        for (unsigned elem = 0; elem < riscv_ame_nelem(env); elem++) {
            unsigned ae = scalar_form ? 0 : ao + elem * a.size;
            unsigned be = bo + elem * b.size;
            unsigned de = old_do + elem * d.size;
            __uint128_t bits;

            if (unary) {
                bits = ame_load_bits(ap, ae, a.size);
                if (integer) {
                    AMEExactInteger value = ame_exact_integer(bits, dtype_a);

                    value.negative = false;
                    bits = ame_exact_convert(value, dtype_d, &saturated);
                } else {
                    float_status work = fs;
                    float128 value;

                    set_float_exception_flags(0, &work);
                    value = float128_abs(ame_math_load(ap, ae, dtype_a,
                                                       &work));
                    bits = ame_math_round(value, dtype_d, &fs);
                    float_raise(get_float_exception_flags(&work), &fs);
                }
            } else if (integer) {
                AMEExactInteger av = ame_exact_integer(
                    ame_load_bits(ap, ae, scalar_form ? b.size : a.size),
                    scalar_form ? dtype_b : dtype_a);
                AMEExactInteger bv = ame_exact_integer(
                    ame_load_bits(bp, be, b.size), dtype_b);
                AMEExactInteger dv = { 0 };

                if (ame_ew_reads_dest(op)) {
                    dv = ame_exact_integer(ame_load_bits(old_dp, de, d.size),
                                           dtype_d);
                }

                bits = ame_exact_convert(
                    ame_ew_integer_value(op, av, bv, dv, dtype_d),
                    dtype_d, &saturated);
            } else {
                float_status work = fs;
                float128 av, bv, dv, value;

                set_float_exception_flags(0, &work);
                set_float_rounding_mode(float_round_to_odd, &work);
                av = ame_math_load(ap, ae, scalar_form ? dtype_b : dtype_a,
                                   &work);
                bv = ame_math_load(bp, be, dtype_b, &work);
                dv = ame_ew_reads_dest(op) ?
                     ame_math_load(old_dp, de, dtype_d, &work) : float128_zero;
                value = ame_ew_float_value(op, av, bv, dv, &work);
                if (float128_is_zero(value) &&
                    (op & ~1) != AME_MMAX && (op & ~1) != AME_MMIN) {
                    float_status sign_status = fs;

                    set_float_exception_flags(0, &sign_status);
                    value = ame_ew_float_value(op, av, bv, dv,
                                               &sign_status);
                }
                bits = ame_math_round(value, dtype_d, &fs);
                float_raise(get_float_exception_flags(&work), &fs);
            }
            ame_store_bits(new_dp, new_do + elem * d.size, d.size, bits);
        }
    }

    memcpy(riscv_ame_mreg(env, md), result + md * riscv_ame_mreg_bytes(env),
           d.span * riscv_ame_mreg_bytes(env));
    if (saturated) {
        env->amexsat |= AMEXSAT_SAT;
    }
    if (floating) {
        ame_commit_float_flags(env, &fs);
    }
}
