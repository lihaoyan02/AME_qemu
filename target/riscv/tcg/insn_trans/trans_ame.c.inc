/*
 * Attached Matrix Extension translation routines.
 */

static bool trans_agettyp(DisasContext *ctx, arg_ame_agettyp *a)
{
    TCGv_i32 datatype;
    TCGv dest;

    if (!ctx->cfg_ptr->ext_xame) {
        return false;
    }
    if (a->ad >= ctx->cfg_ptr->ame_acc_regs) {
        return false;
    }

    gen_helper_ame_check_enabled(tcg_env);

    datatype = tcg_temp_new_i32();
    dest = tcg_temp_new();
    tcg_gen_ld_i32(datatype, tcg_env,
                   offsetof(CPURISCVState, accd) + a->ad * sizeof(uint32_t));
    tcg_gen_extu_i32_tl(dest, datatype);
    gen_set_gpr(ctx, a->rd, dest);
    return true;
}

static bool trans_asettyp(DisasContext *ctx, arg_ame_asettyp *a)
{
    TCGv_ptr acc;
    TCGv_i32 datatype;

    if (!ctx->cfg_ptr->ext_xame) {
        return false;
    }
    if (a->ad >= ctx->cfg_ptr->ame_acc_regs) {
        return false;
    }

    acc = tcg_temp_new_ptr();
    datatype = tcg_temp_new_i32();
    tcg_gen_addi_ptr(acc, tcg_env,
                     offsetof(CPURISCVState, acc) +
                     a->ad * ctx->cfg_ptr->ame_n * ctx->cfg_ptr->ame_n *
                     AME_MAX_DATATYPE_SIZE / 8);
    tcg_gen_trunc_tl_i32(datatype, get_gpr(ctx, a->rs1, EXT_NONE));
    gen_helper_ame_asettyp(acc, datatype, tcg_env);
    return true;
}

static bool trans_mgettyp(DisasContext *ctx, arg_ame_mgettyp *a)
{
    TCGv_i32 datatype;
    TCGv dest;

    if (!ctx->cfg_ptr->ext_xame) {
        return false;
    }
    if (a->ms1 >= ctx->cfg_ptr->ame_m_regs) {
        return false;
    }

    gen_helper_ame_check_enabled(tcg_env);

    datatype = tcg_temp_new_i32();
    dest = tcg_temp_new();
    tcg_gen_ld_i32(datatype, tcg_env,
                   offsetof(CPURISCVState, md) + a->ms1 * sizeof(uint32_t));
    tcg_gen_extu_i32_tl(dest, datatype);
    gen_set_gpr(ctx, a->rd, dest);
    return true;
}

static bool trans_msettyp(DisasContext *ctx, arg_ame_msettyp *a)
{
    TCGv_i32 datatype;

    if (!ctx->cfg_ptr->ext_xame) {
        return false;
    }
    if (a->md >= ctx->cfg_ptr->ame_m_regs) {
        return false;
    }

    datatype = tcg_temp_new_i32();
    tcg_gen_trunc_tl_i32(datatype, get_gpr(ctx, a->rs1, EXT_NONE));
    gen_helper_ame_msettyp(tcg_env, tcg_constant_i32(a->md), datatype);
    return true;
}

enum {
    AME_MEM_1R,
    AME_MEM_CM,
    AME_MEM_RM,
    AME_MEM_ST,
    AME_MEM_TST,
};

static bool trans_ame_mls(DisasContext *ctx, arg_ame_load *a,
                          unsigned layout)
{
    TCGv base;

    if (!ctx->cfg_ptr->ext_xame) {
        return false;
    }
    if (a->md >= ctx->cfg_ptr->ame_m_regs) {
        return false;
    }

    base = get_address(ctx, a->rs1, 0);
    gen_helper_ame_mls(tcg_env, tcg_constant_i32(a->md), base,
                       tcg_constant_i32(layout));
    return true;
}

static bool trans_ame_mss(DisasContext *ctx, arg_ame_load *a,
                          unsigned layout)
{
    TCGv base;

    if (!ctx->cfg_ptr->ext_xame) {
        return false;
    }
    if (a->md >= ctx->cfg_ptr->ame_m_regs) {
        return false;
    }

    base = get_address(ctx, a->rs1, 0);
    gen_helper_ame_mss(tcg_env, tcg_constant_i32(a->md), base,
                       tcg_constant_i32(layout));
    return true;
}

static bool trans_mls_1r(DisasContext *ctx, arg_mls_1r *a)
{
    return trans_ame_mls(ctx, (arg_ame_load *)a, AME_MEM_1R);
}

static bool trans_mls_cm(DisasContext *ctx, arg_mls_cm *a)
{
    return trans_ame_mls(ctx, (arg_ame_load *)a, AME_MEM_CM);
}

static bool trans_mls_rm(DisasContext *ctx, arg_mls_rm *a)
{
    return trans_ame_mls(ctx, (arg_ame_load *)a, AME_MEM_RM);
}

static bool trans_mss_1r(DisasContext *ctx, arg_mss_1r *a)
{
    return trans_ame_mss(ctx, (arg_ame_load *)a, AME_MEM_1R);
}

static bool trans_mss_cm(DisasContext *ctx, arg_mss_cm *a)
{
    return trans_ame_mss(ctx, (arg_ame_load *)a, AME_MEM_CM);
}

static bool trans_mss_rm(DisasContext *ctx, arg_mss_rm *a)
{
    return trans_ame_mss(ctx, (arg_ame_load *)a, AME_MEM_RM);
}

static bool trans_ame_stride(DisasContext *ctx, arg_ame_stride *a,
                             unsigned layout, bool store)
{
    TCGv base;
    TCGv stride;

    if (!ctx->cfg_ptr->ext_xame) {
        return false;
    }
    if (a->md >= ctx->cfg_ptr->ame_m_regs) {
        return false;
    }

    base = get_address(ctx, a->rs1, 0);
    stride = get_gpr(ctx, a->rs2, EXT_NONE);
    if (store) {
        gen_helper_ame_mss_stride(tcg_env, tcg_constant_i32(a->md),
                                  base, stride, tcg_constant_i32(layout));
    } else {
        gen_helper_ame_mls_stride(tcg_env, tcg_constant_i32(a->md),
                                  base, stride, tcg_constant_i32(layout));
    }
    return true;
}

static bool trans_mls_st(DisasContext *ctx, arg_mls_st *a)
{
    return trans_ame_stride(ctx, (arg_ame_stride *)a, AME_MEM_ST, false);
}

static bool trans_mls_tst(DisasContext *ctx, arg_mls_tst *a)
{
    return trans_ame_stride(ctx, (arg_ame_stride *)a, AME_MEM_TST, false);
}

static bool trans_mss_st(DisasContext *ctx, arg_mss_st *a)
{
    return trans_ame_stride(ctx, (arg_ame_stride *)a, AME_MEM_ST, true);
}

static bool trans_mss_tst(DisasContext *ctx, arg_mss_tst *a)
{
    return trans_ame_stride(ctx, (arg_ame_stride *)a, AME_MEM_TST, true);
}

static bool trans_ame_acquire(DisasContext *ctx, arg_ame_acquire *a)
{
    TCGv result;

    if (!ctx->cfg_ptr->ext_xame) {
        return false;
    }

    result = tcg_temp_new();
    gen_helper_ame_acquire(result, tcg_env,
                           get_gpr(ctx, a->rs1, EXT_NONE));
    gen_set_gpr(ctx, a->rd, result);
    return true;
}

static bool trans_ame_release(DisasContext *ctx, arg_ame_release *a)
{
    if (!ctx->cfg_ptr->ext_xame) {
        return false;
    }

    gen_helper_ame_release(tcg_env);
    return true;
}

static bool trans_mzero_2d_acc(DisasContext *ctx, arg_ame_acc *a)
{
    if (!ctx->cfg_ptr->ext_xame) {
        return false;
    }
    if (a->ad >= ctx->cfg_ptr->ame_acc_regs) {
        return false;
    }

    gen_helper_ame_mzero_2d_acc(tcg_env, tcg_constant_i32(a->ad));
    return true;
}

static bool trans_mzero_2d_m(DisasContext *ctx, arg_ame_mreg *a)
{
    if (!ctx->cfg_ptr->ext_xame) {
        return false;
    }
    if (a->md >= ctx->cfg_ptr->ame_m_regs) {
        return false;
    }
    gen_helper_ame_mzero_2d_m(tcg_env, tcg_constant_i32(a->md));
    return true;
}

static bool trans_mmulacc_2d_common(DisasContext *ctx, arg_ame_mmulacc *a)
{
    if (!ctx->cfg_ptr->ext_xame) {
        return false;
    }
    if (a->ms1 >= ctx->cfg_ptr->ame_m_regs ||
        a->ms2 >= ctx->cfg_ptr->ame_m_regs ||
        a->acc >= ctx->cfg_ptr->ame_acc_regs) {
        return false;
    }

    gen_helper_ame_mmulacc_2d(tcg_env, tcg_constant_i32(a->ms1),
                              tcg_constant_i32(a->ms2),
                              tcg_constant_i32(a->acc),
                              tcg_constant_i32((ctx->opcode >> 25) & 0x7f));
    return true;
}

static bool trans_mmulacc_2d(DisasContext *ctx, arg_mmulacc_2d *a)
{
    return trans_mmulacc_2d_common(ctx, (arg_ame_mmulacc *)a);
}

static bool trans_mmulaccneg_2d(DisasContext *ctx, arg_mmulaccneg_2d *a)
{
    return trans_mmulacc_2d_common(ctx, (arg_ame_mmulacc *)a);
}

static bool trans_mmulatacc_2d(DisasContext *ctx, arg_mmulatacc_2d *a)
{
    return trans_mmulacc_2d_common(ctx, (arg_ame_mmulacc *)a);
}

static bool trans_mmulataccneg_2d(DisasContext *ctx,
                                  arg_mmulataccneg_2d *a)
{
    return trans_mmulacc_2d_common(ctx, (arg_ame_mmulacc *)a);
}

static bool trans_mmulbtacc_2d(DisasContext *ctx, arg_mmulbtacc_2d *a)
{
    return trans_mmulacc_2d_common(ctx, (arg_ame_mmulacc *)a);
}

static bool trans_mmulbtaccneg_2d(DisasContext *ctx,
                                  arg_mmulbtaccneg_2d *a)
{
    return trans_mmulacc_2d_common(ctx, (arg_ame_mmulacc *)a);
}

static bool trans_mmov_ma(DisasContext *ctx, arg_ame_mmov_ma *a)
{
    if (!ctx->cfg_ptr->ext_xame) {
        return false;
    }
    if (a->md >= ctx->cfg_ptr->ame_m_regs ||
        a->acc >= ctx->cfg_ptr->ame_acc_regs) {
        return false;
    }

    gen_helper_ame_mmov_ma(tcg_env, tcg_constant_i32(a->md),
                           tcg_constant_i32(a->acc));
    return true;
}

static bool trans_mmov_am(DisasContext *ctx, arg_ame_mmov_am *a)
{
    if (!ctx->cfg_ptr->ext_xame) {
        return false;
    }
    if (a->ms >= ctx->cfg_ptr->ame_m_regs ||
        a->acc >= ctx->cfg_ptr->ame_acc_regs) {
        return false;
    }
    gen_helper_ame_mmov_am(tcg_env, tcg_constant_i32(a->acc),
                           tcg_constant_i32(a->ms));
    return true;
}

static bool trans_mmov_mm(DisasContext *ctx, arg_ame_mmov_mm *a)
{
    if (!ctx->cfg_ptr->ext_xame) {
        return false;
    }
    if (a->md >= ctx->cfg_ptr->ame_m_regs ||
        a->ms >= ctx->cfg_ptr->ame_m_regs) {
        return false;
    }
    gen_helper_ame_mmov_mm(tcg_env, tcg_constant_i32(a->md),
                           tcg_constant_i32(a->ms));
    return true;
}

static bool trans_mcolid_ew(DisasContext *ctx, arg_ame_mreg *a)
{
    if (!ctx->cfg_ptr->ext_xame) {
        return false;
    }
    if (a->md >= ctx->cfg_ptr->ame_m_regs) {
        return false;
    }
    gen_helper_ame_mcolid_ew(tcg_env, tcg_constant_i32(a->md));
    return true;
}

static bool trans_mrowid_ew(DisasContext *ctx, arg_ame_mreg *a)
{
    if (!ctx->cfg_ptr->ext_xame) {
        return false;
    }
    if (a->md >= ctx->cfg_ptr->ame_m_regs) {
        return false;
    }
    gen_helper_ame_mrowid_ew(tcg_env, tcg_constant_i32(a->md));
    return true;
}

static bool trans_mbcast_m_x(DisasContext *ctx, arg_ame_bcast *a)
{
    if (!ctx->cfg_ptr->ext_xame) {
        return false;
    }
    if (a->md >= ctx->cfg_ptr->ame_m_regs) {
        return false;
    }
    gen_helper_ame_mbcast(tcg_env, tcg_constant_i32(a->md),
                          get_gpr(ctx, a->rs1, EXT_NONE),
                          get_gpr(ctx, a->rs2, EXT_NONE));
    return true;
}

static bool trans_mconv_ew(DisasContext *ctx, arg_ame_conv *a)
{
    if (!ctx->cfg_ptr->ext_xame) {
        return false;
    }
    if (a->md >= ctx->cfg_ptr->ame_m_regs ||
        a->rs1 >= ctx->cfg_ptr->ame_m_regs) {
        return false;
    }
    gen_helper_ame_mconv(tcg_env, tcg_constant_i32(a->md),
                         tcg_constant_i32(a->rs1));
    return true;
}

static bool trans_ame_mmove_m_x(DisasContext *ctx, unsigned md,
                                unsigned pos, unsigned scalar, unsigned size)
{
    if (!ctx->cfg_ptr->ext_xame) {
        return false;
    }
    if (md >= ctx->cfg_ptr->ame_m_regs) {
        return false;
    }
    gen_helper_ame_mmove_mx(tcg_env, tcg_constant_i32(md),
                            get_gpr(ctx, pos, EXT_NONE),
                            get_gpr(ctx, scalar, EXT_NONE),
                            tcg_constant_i32(size));
    return true;
}

static bool trans_ame_mmove_x_m(DisasContext *ctx, unsigned rd,
                                unsigned ms, unsigned pos, unsigned size)
{
    if (!ctx->cfg_ptr->ext_xame) {
        return false;
    }
    if (ms >= ctx->cfg_ptr->ame_m_regs) {
        return false;
    }
    TCGv value = tcg_temp_new();

    gen_helper_ame_mmove_xm(value, tcg_env, tcg_constant_i32(ms),
                            get_gpr(ctx, pos, EXT_NONE),
                            tcg_constant_i32(size));
    gen_set_gpr(ctx, rd, value);
    return true;
}

static bool trans_mmove8_m_x(DisasContext *ctx, arg_ame_mconv_debug *a)
{
    return trans_ame_mmove_m_x(ctx, a->md, a->rs2, a->rs1, 8);
}

static bool trans_mmove16_m_x(DisasContext *ctx, arg_ame_mconv_debug *a)
{
    return trans_ame_mmove_m_x(ctx, a->md, a->rs2, a->rs1, 16);
}

static bool trans_mmove32_m_x(DisasContext *ctx, arg_ame_mconv_debug *a)
{
    return trans_ame_mmove_m_x(ctx, a->md, a->rs2, a->rs1, 32);
}

static bool trans_mmove64_m_x(DisasContext *ctx, arg_ame_mconv_debug *a)
{
    return trans_ame_mmove_m_x(ctx, a->md, a->rs2, a->rs1, 64);
}

static bool trans_mmove8_x_m(DisasContext *ctx, arg_ame_xconv_debug *a)
{
    return trans_ame_mmove_x_m(ctx, a->rd, a->rs1, a->rs2, 8);
}

static bool trans_mmove16_x_m(DisasContext *ctx, arg_ame_xconv_debug *a)
{
    return trans_ame_mmove_x_m(ctx, a->rd, a->rs1, a->rs2, 16);
}

static bool trans_mmove32_x_m(DisasContext *ctx, arg_ame_xconv_debug *a)
{
    return trans_ame_mmove_x_m(ctx, a->rd, a->rs1, a->rs2, 32);
}

static bool trans_mmove64_x_m(DisasContext *ctx, arg_ame_xconv_debug *a)
{
    return trans_ame_mmove_x_m(ctx, a->rd, a->rs1, a->rs2, 64);
}

static bool trans_ame_pack(DisasContext *ctx, arg_ame_pack *a, bool unpack)
{
    if (!ctx->cfg_ptr->ext_xame) {
        return false;
    }
    if (a->md >= ctx->cfg_ptr->ame_m_regs ||
        a->rs2 >= ctx->cfg_ptr->ame_m_regs) {
        return false;
    }
    if (unpack) {
        gen_helper_ame_munpack(tcg_env, tcg_constant_i32(a->md),
                               get_gpr(ctx, a->rs1, EXT_NONE),
                               tcg_constant_i32(a->rs2));
    } else {
        gen_helper_ame_mpack(tcg_env, tcg_constant_i32(a->md),
                             get_gpr(ctx, a->rs1, EXT_NONE),
                             tcg_constant_i32(a->rs2));
    }
    return true;
}

static bool trans_mpack_ew_x(DisasContext *ctx, arg_ame_pack *a)
{
    return trans_ame_pack(ctx, a, false);
}

static bool trans_munpack_ew_x(DisasContext *ctx, arg_ame_pack *a)
{
    return trans_ame_pack(ctx, a, true);
}

static bool trans_ame_bitwise_ew(DisasContext *ctx, unsigned md,
                                 unsigned ms1, unsigned ms2)
{
    if (!ctx->cfg_ptr->ext_xame) {
        return false;
    }
    if (md >= ctx->cfg_ptr->ame_m_regs || ms1 >= ctx->cfg_ptr->ame_m_regs ||
        ms2 >= ctx->cfg_ptr->ame_m_regs) {
        return false;
    }
    gen_helper_ame_mbitwise(tcg_env, tcg_constant_i32(md),
                            tcg_constant_i32(ms1),
                            tcg_constant_i32(ms2), tcg_constant_tl(0),
                            tcg_constant_i32((ctx->opcode >> 25) & 0x7f));
    return true;
}

static bool trans_ame_arith_ew(DisasContext *ctx, unsigned md,
                               unsigned ms1, unsigned ms2)
{
    if (!ctx->cfg_ptr->ext_xame || md >= ctx->cfg_ptr->ame_m_regs ||
        ms1 >= ctx->cfg_ptr->ame_m_regs || ms2 >= ctx->cfg_ptr->ame_m_regs) {
        return false;
    }
    gen_helper_ame_ew_arith(
        tcg_env, tcg_constant_i32(md), tcg_constant_i32(ms1),
        tcg_constant_i32(ms2), tcg_constant_tl(0),
        tcg_constant_i32((ctx->opcode >> 25) & 0x7f));
    return true;
}

static bool trans_ame_arith_ew_x(DisasContext *ctx, unsigned md,
                                 unsigned rs1, unsigned ms2)
{
    if (!ctx->cfg_ptr->ext_xame || md >= ctx->cfg_ptr->ame_m_regs ||
        ms2 >= ctx->cfg_ptr->ame_m_regs) {
        return false;
    }
    gen_helper_ame_ew_arith(
        tcg_env, tcg_constant_i32(md), tcg_constant_i32(0),
        tcg_constant_i32(ms2), get_gpr(ctx, rs1, EXT_NONE),
        tcg_constant_i32((ctx->opcode >> 25) & 0x7f));
    return true;
}

static bool trans_mabs_ew(DisasContext *ctx, arg_mabs_ew *a)
{
    if (!ctx->cfg_ptr->ext_xame || a->md >= ctx->cfg_ptr->ame_m_regs ||
        a->rs1 >= ctx->cfg_ptr->ame_m_regs) {
        return false;
    }
    gen_helper_ame_ew_arith(
        tcg_env, tcg_constant_i32(a->md), tcg_constant_i32(a->rs1),
        tcg_constant_i32(0), tcg_constant_tl(0),
        tcg_constant_i32(ctx->opcode >> 20));
    return true;
}

#define TRANS_AME_ARITH_EW(NAME)                                          \
    static bool trans_##NAME(DisasContext *ctx, arg_##NAME *a)             \
    {                                                                       \
        return trans_ame_arith_ew(ctx, a->md, a->ms1, a->ms2);             \
    }

#define TRANS_AME_ARITH_EW_X(NAME)                                        \
    static bool trans_##NAME(DisasContext *ctx, arg_##NAME *a)             \
    {                                                                       \
        return trans_ame_arith_ew_x(ctx, a->md, a->rs1, a->ms2);           \
    }

TRANS_AME_ARITH_EW(mabsdiff_ew)
TRANS_AME_ARITH_EW_X(mabsdiff_ew_x)
TRANS_AME_ARITH_EW(madd_ew)
TRANS_AME_ARITH_EW_X(madd_ew_x)
TRANS_AME_ARITH_EW(mhdiff_ew)
TRANS_AME_ARITH_EW_X(mhdiff_ew_x)
TRANS_AME_ARITH_EW(mmax_ew)
TRANS_AME_ARITH_EW_X(mmax_ew_x)
TRANS_AME_ARITH_EW(mmean_ew)
TRANS_AME_ARITH_EW_X(mmean_ew_x)
TRANS_AME_ARITH_EW(mmin_ew)
TRANS_AME_ARITH_EW_X(mmin_ew_x)
TRANS_AME_ARITH_EW(mmul_ew)
TRANS_AME_ARITH_EW_X(mmul_ew_x)
TRANS_AME_ARITH_EW(mmulacc_ew)
TRANS_AME_ARITH_EW_X(mmulacc_ew_x)
TRANS_AME_ARITH_EW(mmulaccneg_ew)
TRANS_AME_ARITH_EW_X(mmulaccneg_ew_x)
TRANS_AME_ARITH_EW(mmuladd_ew)
TRANS_AME_ARITH_EW_X(mmuladd_ew_x)
TRANS_AME_ARITH_EW(mmulneg_ew)
TRANS_AME_ARITH_EW_X(mmulneg_ew_x)
TRANS_AME_ARITH_EW(mmulsub_ew)
TRANS_AME_ARITH_EW_X(mmulsub_ew_x)
TRANS_AME_ARITH_EW(msub_ew)
TRANS_AME_ARITH_EW_X(msub_ew_x)

#undef TRANS_AME_ARITH_EW
#undef TRANS_AME_ARITH_EW_X

static bool trans_ame_bitwise_ew_x(DisasContext *ctx, unsigned md,
                                   unsigned rs1, unsigned ms2)
{
    if (!ctx->cfg_ptr->ext_xame) {
        return false;
    }
    if (md >= ctx->cfg_ptr->ame_m_regs || ms2 >= ctx->cfg_ptr->ame_m_regs) {
        return false;
    }
    gen_helper_ame_mbitwise(tcg_env, tcg_constant_i32(md),
                            tcg_constant_i32(0), tcg_constant_i32(ms2),
                            get_gpr(ctx, rs1, EXT_NONE),
                            tcg_constant_i32((ctx->opcode >> 25) & 0x7f));
    return true;
}

static bool trans_mand_ew(DisasContext *ctx, arg_mand_ew *a)
{
    return trans_ame_bitwise_ew(ctx, a->md, a->ms1, a->ms2);
}

static bool trans_mand_ew_x(DisasContext *ctx, arg_mand_ew_x *a)
{
    return trans_ame_bitwise_ew_x(ctx, a->md, a->rs1, a->ms2);
}

static bool trans_mandnot_ew(DisasContext *ctx, arg_mandnot_ew *a)
{
    return trans_ame_bitwise_ew(ctx, a->md, a->ms1, a->ms2);
}

static bool trans_mandnot_ew_x(DisasContext *ctx, arg_mandnot_ew_x *a)
{
    return trans_ame_bitwise_ew_x(ctx, a->md, a->rs1, a->ms2);
}

static bool trans_mor_ew(DisasContext *ctx, arg_mor_ew *a)
{
    return trans_ame_bitwise_ew(ctx, a->md, a->ms1, a->ms2);
}

static bool trans_mor_ew_x(DisasContext *ctx, arg_mor_ew_x *a)
{
    return trans_ame_bitwise_ew_x(ctx, a->md, a->rs1, a->ms2);
}

static bool trans_mornot_ew(DisasContext *ctx, arg_mornot_ew *a)
{
    return trans_ame_bitwise_ew(ctx, a->md, a->ms1, a->ms2);
}

static bool trans_mornot_ew_x(DisasContext *ctx, arg_mornot_ew_x *a)
{
    return trans_ame_bitwise_ew_x(ctx, a->md, a->rs1, a->ms2);
}

static bool trans_msll_ew(DisasContext *ctx, arg_msll_ew *a)
{
    return trans_ame_bitwise_ew(ctx, a->md, a->ms1, a->ms2);
}

static bool trans_msll_ew_x(DisasContext *ctx, arg_msll_ew_x *a)
{
    return trans_ame_bitwise_ew_x(ctx, a->md, a->rs1, a->ms2);
}

static bool trans_msra_ew(DisasContext *ctx, arg_msra_ew *a)
{
    return trans_ame_bitwise_ew(ctx, a->md, a->ms1, a->ms2);
}

static bool trans_msra_ew_x(DisasContext *ctx, arg_msra_ew_x *a)
{
    return trans_ame_bitwise_ew_x(ctx, a->md, a->rs1, a->ms2);
}

static bool trans_msrl_ew(DisasContext *ctx, arg_msrl_ew *a)
{
    return trans_ame_bitwise_ew(ctx, a->md, a->ms1, a->ms2);
}

static bool trans_msrl_ew_x(DisasContext *ctx, arg_msrl_ew_x *a)
{
    return trans_ame_bitwise_ew_x(ctx, a->md, a->rs1, a->ms2);
}

static bool trans_mxor_ew(DisasContext *ctx, arg_mxor_ew *a)
{
    return trans_ame_bitwise_ew(ctx, a->md, a->ms1, a->ms2);
}

static bool trans_mxor_ew_x(DisasContext *ctx, arg_mxor_ew_x *a)
{
    return trans_ame_bitwise_ew_x(ctx, a->md, a->rs1, a->ms2);
}

static bool trans_ame_compare_select_ew(DisasContext *ctx, unsigned md,
                                        unsigned ms1, unsigned ms2)
{
    if (!ctx->cfg_ptr->ext_xame) {
        return false;
    }
    if (md >= ctx->cfg_ptr->ame_m_regs || ms1 >= ctx->cfg_ptr->ame_m_regs ||
        ms2 >= ctx->cfg_ptr->ame_m_regs) {
        return false;
    }
    gen_helper_ame_mcompare_select(
        tcg_env, tcg_constant_i32(md), tcg_constant_i32(ms1),
        tcg_constant_i32(ms2), tcg_constant_tl(0),
        tcg_constant_i32((ctx->opcode >> 25) & 0x7f));
    return true;
}

static bool trans_ame_compare_select_ew_x(DisasContext *ctx, unsigned md,
                                          unsigned rs1, unsigned ms2)
{
    if (!ctx->cfg_ptr->ext_xame) {
        return false;
    }
    if (md >= ctx->cfg_ptr->ame_m_regs || ms2 >= ctx->cfg_ptr->ame_m_regs) {
        return false;
    }
    gen_helper_ame_mcompare_select(
        tcg_env, tcg_constant_i32(md), tcg_constant_i32(0),
        tcg_constant_i32(ms2), get_gpr(ctx, rs1, EXT_NONE),
        tcg_constant_i32((ctx->opcode >> 25) & 0x7f));
    return true;
}

static bool trans_mcmovge_ew(DisasContext *ctx, arg_mcmovge_ew *a)
{
    return trans_ame_compare_select_ew(ctx, a->md, a->ms1, a->ms2);
}

static bool trans_mcmovlt_ew(DisasContext *ctx, arg_mcmovlt_ew *a)
{
    return trans_ame_compare_select_ew(ctx, a->md, a->ms1, a->ms2);
}

static bool trans_mcmpge_ew(DisasContext *ctx, arg_mcmpge_ew *a)
{
    return trans_ame_compare_select_ew(ctx, a->md, a->ms1, a->ms2);
}

static bool trans_mcmpge_ew_x(DisasContext *ctx, arg_mcmpge_ew_x *a)
{
    return trans_ame_compare_select_ew_x(ctx, a->md, a->rs1, a->ms2);
}

static bool trans_mcmplt_ew(DisasContext *ctx, arg_mcmplt_ew *a)
{
    return trans_ame_compare_select_ew(ctx, a->md, a->ms1, a->ms2);
}

static bool trans_mcmplt_ew_x(DisasContext *ctx, arg_mcmplt_ew_x *a)
{
    return trans_ame_compare_select_ew_x(ctx, a->md, a->rs1, a->ms2);
}

static bool trans_mselge_ew(DisasContext *ctx, arg_mselge_ew *a)
{
    return trans_ame_compare_select_ew(ctx, a->md, a->ms1, a->ms2);
}

static bool trans_msellt_ew(DisasContext *ctx, arg_msellt_ew *a)
{
    return trans_ame_compare_select_ew(ctx, a->md, a->ms1, a->ms2);
}

static bool trans_ame_rowcol_ew(DisasContext *ctx, unsigned md,
                                unsigned ms1, unsigned ms2)
{
    if (!ctx->cfg_ptr->ext_xame) {
        return false;
    }
    if (md >= ctx->cfg_ptr->ame_m_regs || ms1 >= ctx->cfg_ptr->ame_m_regs ||
        ms2 >= ctx->cfg_ptr->ame_m_regs) {
        return false;
    }
    gen_helper_ame_mrowcol(
        tcg_env, tcg_constant_i32(md), tcg_constant_i32(ms1),
        tcg_constant_i32(ms2), tcg_constant_tl(0),
        tcg_constant_i32((ctx->opcode >> 25) & 0x7f));
    return true;
}

static bool trans_ame_rowcol_ew_x(DisasContext *ctx, unsigned md,
                                  unsigned rs1, unsigned ms2)
{
    if (!ctx->cfg_ptr->ext_xame) {
        return false;
    }
    if (md >= ctx->cfg_ptr->ame_m_regs || ms2 >= ctx->cfg_ptr->ame_m_regs) {
        return false;
    }
    gen_helper_ame_mrowcol(
        tcg_env, tcg_constant_i32(md), tcg_constant_i32(0),
        tcg_constant_i32(ms2), get_gpr(ctx, rs1, EXT_NONE),
        tcg_constant_i32((ctx->opcode >> 25) & 0x7f));
    return true;
}

static bool trans_ame_rowcol_zip(DisasContext *ctx, unsigned ms1,
                                 unsigned ms2)
{
    if (!ctx->cfg_ptr->ext_xame) {
        return false;
    }
    if (ms1 >= ctx->cfg_ptr->ame_m_regs || ms2 >= ctx->cfg_ptr->ame_m_regs ||
        ms1 == ms2) {
        return false;
    }
    gen_helper_ame_mrowcol_zip(
        tcg_env, tcg_constant_i32(ms1), tcg_constant_i32(ms2),
        tcg_constant_i32((ctx->opcode >> 25) & 0x7f));
    return true;
}

#define TRANS_AME_ROWCOL_EW(NAME)                                           \
    static bool trans_##NAME(DisasContext *ctx, arg_##NAME *a)              \
    {                                                                        \
        return trans_ame_rowcol_ew(ctx, a->md, a->ms1, a->ms2);             \
    }

#define TRANS_AME_ROWCOL_EW_X(NAME)                                         \
    static bool trans_##NAME(DisasContext *ctx, arg_##NAME *a)              \
    {                                                                        \
        return trans_ame_rowcol_ew_x(ctx, a->md, a->rs1, a->ms2);           \
    }

#define TRANS_AME_ROWCOL_ZIP(NAME)                                          \
    static bool trans_##NAME(DisasContext *ctx, arg_##NAME *a)              \
    {                                                                        \
        return trans_ame_rowcol_zip(ctx, a->ms1, a->ms2);                   \
    }

TRANS_AME_ROWCOL_EW_X(mcolbcast_ew_x)
TRANS_AME_ROWCOL_EW(mcolgather_ew)
TRANS_AME_ROWCOL_EW_X(mcolshift_ew_x)
TRANS_AME_ROWCOL_EW_X(mrowbcast_ew_x)
TRANS_AME_ROWCOL_EW(mrowgather_ew)
TRANS_AME_ROWCOL_EW_X(mrowshift_ew_x)
TRANS_AME_ROWCOL_EW(mrowscatadd_ew)
TRANS_AME_ROWCOL_EW(mcolscatadd_ew)
TRANS_AME_ROWCOL_EW(mrowscatmax_ew)
TRANS_AME_ROWCOL_EW(mcolscatmax_ew)
TRANS_AME_ROWCOL_ZIP(mcolunzip_ew)
TRANS_AME_ROWCOL_ZIP(mcolzip_ew)
TRANS_AME_ROWCOL_ZIP(mrowunzip_ew)
TRANS_AME_ROWCOL_ZIP(mrowzip_ew)

#undef TRANS_AME_ROWCOL_EW
#undef TRANS_AME_ROWCOL_EW_X
#undef TRANS_AME_ROWCOL_ZIP

static bool trans_ame_reduction(DisasContext *ctx, arg_ame_conv *a)
{
    if (!ctx->cfg_ptr->ext_xame || a->md >= ctx->cfg_ptr->ame_m_regs ||
        a->rs1 >= ctx->cfg_ptr->ame_m_regs) {
        return false;
    }
    /* The unary selector occupies bits 31:20, not just funct7. */
    gen_helper_ame_reduction(tcg_env, tcg_constant_i32(a->md),
                            tcg_constant_i32(a->rs1),
                            tcg_constant_i32(ctx->opcode >> 20));
    return true;
}

#define TRANS_AME_REDUCTION(NAME)                                      \
    static bool trans_##NAME(DisasContext *ctx, arg_##NAME *a)          \
    {                                                                 \
        return trans_ame_reduction(ctx, a);                            \
    }

TRANS_AME_REDUCTION(mprefixadd_col)
TRANS_AME_REDUCTION(mprefixadd_row)
TRANS_AME_REDUCTION(mprefixmax_col)
TRANS_AME_REDUCTION(mprefixmax_row)
TRANS_AME_REDUCTION(mreduceadd_col)
TRANS_AME_REDUCTION(mreduceadd_row)
TRANS_AME_REDUCTION(mreducemax_col)
TRANS_AME_REDUCTION(mreducemax_row)
TRANS_AME_REDUCTION(mreducemin_col)
TRANS_AME_REDUCTION(mreducemin_row)

#undef TRANS_AME_REDUCTION

static bool trans_ame_math(DisasContext *ctx, unsigned md, unsigned ms1,
                           unsigned ms2, bool scalar, bool unary)
{
    if (!ctx->cfg_ptr->ext_xame || md >= ctx->cfg_ptr->ame_m_regs ||
        (!scalar && ms1 >= ctx->cfg_ptr->ame_m_regs) ||
        (!unary && ms2 >= ctx->cfg_ptr->ame_m_regs)) {
        return false;
    }
    gen_helper_ame_math(tcg_env, tcg_constant_i32(md),
                        tcg_constant_i32(scalar ? 0 : ms1),
                        tcg_constant_i32(ms2),
                        scalar ? get_gpr(ctx, ms1, EXT_NONE) : tcg_constant_tl(0),
                        tcg_constant_i32(unary ? ctx->opcode >> 20 :
                                                 ctx->opcode >> 25));
    return true;
}

#define TRANS_AME_MATH_UNARY(NAME)                                    \
    static bool trans_##NAME(DisasContext *ctx, arg_##NAME *a)         \
    {                                                                \
        return trans_ame_math(ctx, a->md, a->rs1, 0, false, true);     \
    }
#define TRANS_AME_MATH_EW(NAME)                                       \
    static bool trans_##NAME(DisasContext *ctx, arg_##NAME *a)         \
    {                                                                \
        return trans_ame_math(ctx, a->md, a->ms1, a->ms2, false, false); \
    }
#define TRANS_AME_MATH_X(NAME)                                        \
    static bool trans_##NAME(DisasContext *ctx, arg_##NAME *a)         \
    {                                                                \
        return trans_ame_math(ctx, a->md, a->rs1, a->ms2, true, false); \
    }

TRANS_AME_MATH_UNARY(mrec_ew)
TRANS_AME_MATH_UNARY(mrsqrt_ew)
TRANS_AME_MATH_UNARY(msqrt_ew)
TRANS_AME_MATH_UNARY(mcos_ew)
TRANS_AME_MATH_UNARY(mexp2_ew)
TRANS_AME_MATH_UNARY(mlog2_ew)
TRANS_AME_MATH_UNARY(msin_ew)
TRANS_AME_MATH_UNARY(mtanh_ew)
TRANS_AME_MATH_UNARY(mfrintm_ew)
TRANS_AME_MATH_UNARY(mfrintn_ew)
TRANS_AME_MATH_UNARY(mfrintp_ew)
TRANS_AME_MATH_UNARY(mfrintz_ew)
TRANS_AME_MATH_EW(mldexp_ew)
TRANS_AME_MATH_X(mldexp_ew_x)
TRANS_AME_MATH_EW(mldexpacc_ew)
TRANS_AME_MATH_X(mldexpacc_ew_x)
TRANS_AME_MATH_EW(mrdexp_ew)
TRANS_AME_MATH_EW(mrdexpacc_ew)
TRANS_AME_MATH_EW(mlog2sub_ew)
TRANS_AME_MATH_X(mlog2sub_ew_x)
TRANS_AME_MATH_EW(msublog2_ew)
TRANS_AME_MATH_X(msublog2_ew_x)

#undef TRANS_AME_MATH_UNARY
#undef TRANS_AME_MATH_EW
#undef TRANS_AME_MATH_X
