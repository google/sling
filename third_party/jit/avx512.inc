// Auto-generated from Intel instruction tables.
void vaddpd(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0x58, dst, src1, src2, 0, mask, EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W1 | evex_round(er));
}
void vaddpd(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x58, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W1);
}
void vaddps(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0x58, dst, src1, src2, 0, mask, EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_W0 | evex_round(er));
}
void vaddps(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x58, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_W0);
}
void vaddsd(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0x58, dst, src1, src2, 0, mask, EVEX_DT8 | EVEX_ENDS | EVEX_LIG | EVEX_M0F | EVEX_PF2 | EVEX_W1 | evex_round(er));
}
void vaddsd(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x58, dst, src1, src2, 0, mask, EVEX_DT8 | EVEX_ENDS | EVEX_LIG | EVEX_M0F | EVEX_PF2 | EVEX_W1);
}
void vaddss(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0x58, dst, src1, src2, 0, mask, EVEX_DT4 | EVEX_ENDS | EVEX_LIG | EVEX_M0F | EVEX_PF3 | EVEX_W0 | evex_round(er));
}
void vaddss(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x58, dst, src1, src2, 0, mask, EVEX_DT4 | EVEX_ENDS | EVEX_LIG | EVEX_M0F | EVEX_PF3 | EVEX_W0);
}
void valignd(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, int8_t imm8, Mask mask = nomask) {
  zinstr(0x03, dst, src1, src2, imm8, mask, EVEX_BT4 | EVEX_ENDS | EVEX_IMM | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F3A | EVEX_P66 | EVEX_W0);
}
void valignd(ZMMRegister dst, ZMMRegister src1, const Operand &src2, int8_t imm8, Mask mask = nomask) {
  zinstr(0x03, dst, src1, src2, imm8, mask, EVEX_BCST | EVEX_BT4 | EVEX_ENDS | EVEX_IMM | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F3A | EVEX_P66 | EVEX_W0);
}
void valignq(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, int8_t imm8, Mask mask = nomask) {
  zinstr(0x03, dst, src1, src2, imm8, mask, EVEX_BT8 | EVEX_ENDS | EVEX_IMM | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F3A | EVEX_P66 | EVEX_W1);
}
void valignq(ZMMRegister dst, ZMMRegister src1, const Operand &src2, int8_t imm8, Mask mask = nomask) {
  zinstr(0x03, dst, src1, src2, imm8, mask, EVEX_BCST | EVEX_BT8 | EVEX_ENDS | EVEX_IMM | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F3A | EVEX_P66 | EVEX_W1);
}
void vandnpd(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0x55, dst, src1, src2, 0, mask, EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W1);
}
void vandnpd(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x55, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W1);
}
void vandnps(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0x55, dst, src1, src2, 0, mask, EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_W0);
}
void vandnps(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x55, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_W0);
}
void vandpd(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0x54, dst, src1, src2, 0, mask, EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W1);
}
void vandpd(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x54, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W1);
}
void vandps(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0x54, dst, src1, src2, 0, mask, EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_W0);
}
void vandps(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x54, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_W0);
}
void vblendmpd(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0x65, dst, src1, src2, 0, mask, EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vblendmpd(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x65, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vblendmps(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0x65, dst, src1, src2, 0, mask, EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vblendmps(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x65, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vbroadcastf32x4(ZMMRegister dst, const Operand &src, Mask mask = nomask) {
  zinstr(0x1A, dst, src, 0, mask, EVEX_DT16 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vbroadcastf64x4(ZMMRegister dst, const Operand &src, Mask mask = nomask) {
  zinstr(0x1B, dst, src, 0, mask, EVEX_DT32 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vbroadcasti32x4(ZMMRegister dst, const Operand &src, Mask mask = nomask) {
  zinstr(0x5A, dst, src, 0, mask, EVEX_DT16 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vbroadcasti64x4(ZMMRegister dst, const Operand &src, Mask mask = nomask) {
  zinstr(0x5B, dst, src, 0, mask, EVEX_DT32 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vbroadcastsd(ZMMRegister dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x19, dst, src, 0, mask, EVEX_DT8 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vbroadcastsd(ZMMRegister dst, const Operand &src, Mask mask = nomask) {
  zinstr(0x19, dst, src, 0, mask, EVEX_DT8 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vbroadcastss(ZMMRegister dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x18, dst, src, 0, mask, EVEX_DT4 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vbroadcastss(ZMMRegister dst, const Operand &src, Mask mask = nomask) {
  zinstr(0x18, dst, src, 0, mask, EVEX_DT4 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vcmppd(OpmaskRegister dst, ZMMRegister src1, ZMMRegister src2, int8_t imm8, Mask mask = nomask) {
  zinstr(0xC2, dst, src1, src2, imm8, mask, EVEX_BT8 | EVEX_ENDS | EVEX_IMM | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_SAE | EVEX_W1);
}
void vcmppd(OpmaskRegister dst, ZMMRegister src1, const Operand &src2, int8_t imm8, Mask mask = nomask) {
  zinstr(0xC2, dst, src1, src2, imm8, mask, EVEX_BCST | EVEX_BT8 | EVEX_ENDS | EVEX_IMM | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W1);
}
void vcmpps(OpmaskRegister dst, ZMMRegister src1, ZMMRegister src2, int8_t imm8, Mask mask = nomask) {
  zinstr(0xC2, dst, src1, src2, imm8, mask, EVEX_BT4 | EVEX_ENDS | EVEX_IMM | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_SAE | EVEX_W0);
}
void vcmpps(OpmaskRegister dst, ZMMRegister src1, const Operand &src2, int8_t imm8, Mask mask = nomask) {
  zinstr(0xC2, dst, src1, src2, imm8, mask, EVEX_BCST | EVEX_BT4 | EVEX_ENDS | EVEX_IMM | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_W0);
}
void vcmpsd(OpmaskRegister dst, ZMMRegister src1, ZMMRegister src2, int8_t imm8, Mask mask = nomask) {
  zinstr(0xC2, dst, src1, src2, imm8, mask, EVEX_DT8 | EVEX_ENDS | EVEX_IMM | EVEX_LIG | EVEX_M0F | EVEX_PF2 | EVEX_SAE | EVEX_W1);
}
void vcmpsd(OpmaskRegister dst, ZMMRegister src1, const Operand &src2, int8_t imm8, Mask mask = nomask) {
  zinstr(0xC2, dst, src1, src2, imm8, mask, EVEX_DT8 | EVEX_ENDS | EVEX_IMM | EVEX_LIG | EVEX_M0F | EVEX_PF2 | EVEX_SAE | EVEX_W1);
}
void vcmpss(OpmaskRegister dst, ZMMRegister src1, ZMMRegister src2, int8_t imm8, Mask mask = nomask) {
  zinstr(0xC2, dst, src1, src2, imm8, mask, EVEX_DT4 | EVEX_ENDS | EVEX_IMM | EVEX_LIG | EVEX_M0F | EVEX_PF3 | EVEX_SAE | EVEX_W0);
}
void vcmpss(OpmaskRegister dst, ZMMRegister src1, const Operand &src2, int8_t imm8, Mask mask = nomask) {
  zinstr(0xC2, dst, src1, src2, imm8, mask, EVEX_DT4 | EVEX_ENDS | EVEX_IMM | EVEX_LIG | EVEX_M0F | EVEX_PF3 | EVEX_SAE | EVEX_W0);
}
void vcomisd(ZMMRegister dst, ZMMRegister src) {
  zinstr(0x2F, dst, src, 0, nomask, EVEX_DT8 | EVEX_LIG | EVEX_M0F | EVEX_P66 | EVEX_SAE | EVEX_W1);
}
void vcomisd(ZMMRegister dst, const Operand &src) {
  zinstr(0x2F, dst, src, 0, nomask, EVEX_DT8 | EVEX_LIG | EVEX_M0F | EVEX_P66 | EVEX_SAE | EVEX_W1);
}
void vcomiss(ZMMRegister dst, ZMMRegister src) {
  zinstr(0x2F, dst, src, 0, nomask, EVEX_DT4 | EVEX_LIG | EVEX_M0F | EVEX_SAE | EVEX_W0);
}
void vcomiss(ZMMRegister dst, const Operand &src) {
  zinstr(0x2F, dst, src, 0, nomask, EVEX_DT4 | EVEX_LIG | EVEX_M0F | EVEX_SAE | EVEX_W0);
}
void vcompresspd(ZMMRegister dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x8A, dst, src, 0, mask, EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vcompresspd(const Operand &dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x8A, dst, src, 0, mask, EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vcompressps(ZMMRegister dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x8A, dst, src, 0, mask, EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vcompressps(const Operand &dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x8A, dst, src, 0, mask, EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vcvtdq2pd(ZMMRegister dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0xE6, dst, src, 0, mask, EVEX_BT4 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_PF3 | EVEX_W0);
}
void vcvtdq2pd(ZMMRegister dst, const Operand &src, Mask mask = nomask) {
  zinstr(0xE6, dst, src, 0, mask, EVEX_BCST | EVEX_BT4 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_PF3 | EVEX_W0);
}
void vcvtdq2ps(ZMMRegister dst, ZMMRegister src, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0x5B, dst, src, 0, mask, EVEX_BT4 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_W0 | evex_round(er));
}
void vcvtdq2ps(ZMMRegister dst, const Operand &src, Mask mask = nomask) {
  zinstr(0x5B, dst, src, 0, mask, EVEX_BCST | EVEX_BT4 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_W0);
}
void vcvtpd2dq(ZMMRegister dst, ZMMRegister src, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0xE6, dst, src, 0, mask, EVEX_BT8 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_PF2 | EVEX_W1 | evex_round(er));
}
void vcvtpd2dq(ZMMRegister dst, const Operand &src, Mask mask = nomask) {
  zinstr(0xE6, dst, src, 0, mask, EVEX_BCST | EVEX_BT8 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_PF2 | EVEX_W1);
}
void vcvtpd2ps(ZMMRegister dst, ZMMRegister src, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0x5A, dst, src, 0, mask, EVEX_BT8 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W1 | evex_round(er));
}
void vcvtpd2ps(ZMMRegister dst, const Operand &src, Mask mask = nomask) {
  zinstr(0x5A, dst, src, 0, mask, EVEX_BCST | EVEX_BT8 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W1);
}
void vcvtpd2udq(ZMMRegister dst, ZMMRegister src, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0x79, dst, src, 0, mask, EVEX_BT8 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_W1 | evex_round(er));
}
void vcvtpd2udq(ZMMRegister dst, const Operand &src, Mask mask = nomask) {
  zinstr(0x79, dst, src, 0, mask, EVEX_BCST | EVEX_BT8 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_W1);
}
void vcvtph2ps(ZMMRegister dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x13, dst, src, 0, mask, EVEX_DT8 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_SAE | EVEX_W0);
}
void vcvtph2ps(ZMMRegister dst, const Operand &src, Mask mask = nomask) {
  zinstr(0x13, dst, src, 0, mask, EVEX_DT8 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_SAE | EVEX_W0);
}
void vcvtps2dq(ZMMRegister dst, ZMMRegister src, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0x5B, dst, src, 0, mask, EVEX_BT4 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W0 | evex_round(er));
}
void vcvtps2dq(ZMMRegister dst, const Operand &src, Mask mask = nomask) {
  zinstr(0x5B, dst, src, 0, mask, EVEX_BCST | EVEX_BT4 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W0);
}
void vcvtps2pd(ZMMRegister dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x5A, dst, src, 0, mask, EVEX_BT4 | EVEX_DT8 | EVEX_L128 | EVEX_L512 | EVEX_M0F | EVEX_SAE | EVEX_W0);
}
void vcvtps2pd(ZMMRegister dst, const Operand &src, Mask mask = nomask) {
  zinstr(0x5A, dst, src, 0, mask, EVEX_BCST | EVEX_BT4 | EVEX_DT8 | EVEX_L128 | EVEX_L512 | EVEX_M0F | EVEX_W0);
}
void vcvtps2ph(ZMMRegister dst, ZMMRegister src, int8_t imm8, Mask mask = nomask) {
  zinstr(0x1D, dst, src, imm8, mask, EVEX_DT8 | EVEX_IMM | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F3A | EVEX_P66 | EVEX_SAE | EVEX_W0);
}
void vcvtps2ph(const Operand &dst, ZMMRegister src, int8_t imm8, Mask mask = nomask) {
  zinstr(0x1D, dst, src, imm8, mask, EVEX_DT8 | EVEX_IMM | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F3A | EVEX_P66 | EVEX_SAE | EVEX_W0);
}
void vcvtps2udq(ZMMRegister dst, ZMMRegister src, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0x79, dst, src, 0, mask, EVEX_BT4 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_W0 | evex_round(er));
}
void vcvtps2udq(ZMMRegister dst, const Operand &src, Mask mask = nomask) {
  zinstr(0x79, dst, src, 0, mask, EVEX_BCST | EVEX_BT4 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_W0);
}
void vcvtqq2pd(ZMMRegister dst, ZMMRegister src, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0xE6, dst, src, 0, mask, EVEX_BT8 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_PF3 | EVEX_W1 | evex_round(er));
}
void vcvtqq2pd(ZMMRegister dst, const Operand &src, Mask mask = nomask) {
  zinstr(0xE6, dst, src, 0, mask, EVEX_BCST | EVEX_BT8 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_PF3 | EVEX_W1);
}
void vcvtsd2si(Register dst, ZMMRegister src, RoundingMode er = noround) {
  zinstr(0x2D, dst, src, 0, nomask, EVEX_DT8 | EVEX_LIG | EVEX_M0F | EVEX_PF2 | EVEX_W0 | EVEX_W1 | evex_round(er));
}
void vcvtsd2si(Register dst, const Operand &src) {
  zinstr(0x2D, dst, src, 0, nomask, EVEX_DT8 | EVEX_LIG | EVEX_M0F | EVEX_PF2 | EVEX_W0 | EVEX_W1);
}
void vcvtsd2ss(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0x5A, dst, src1, src2, 0, mask, EVEX_DT8 | EVEX_ENDS | EVEX_LIG | EVEX_M0F | EVEX_PF2 | EVEX_W1 | evex_round(er));
}
void vcvtsd2ss(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x5A, dst, src1, src2, 0, mask, EVEX_DT8 | EVEX_ENDS | EVEX_LIG | EVEX_M0F | EVEX_PF2 | EVEX_W1);
}
void vcvtsd2usi(Register dst, ZMMRegister src, RoundingMode er = noround) {
  zinstr(0x79, dst, src, 0, nomask, EVEX_DT8 | EVEX_LIG | EVEX_M0F | EVEX_PF2 | EVEX_W0 | EVEX_W1 | evex_round(er));
}
void vcvtsd2usi(Register dst, const Operand &src) {
  zinstr(0x79, dst, src, 0, nomask, EVEX_DT8 | EVEX_LIG | EVEX_M0F | EVEX_PF2 | EVEX_W0 | EVEX_W1);
}
void vcvtsi2sd(ZMMRegister dst, ZMMRegister src1, Register src2, RoundingMode er = noround) {
  zinstr(0x2A, dst, src1, src2, 0, nomask, EVEX_DT4 | EVEX_DT8 | EVEX_ENDS | EVEX_LIG | EVEX_M0F | EVEX_PF2 | EVEX_W0 | EVEX_W1 | evex_round(er));
}
void vcvtsi2sd(ZMMRegister dst, ZMMRegister src1, const Operand &src2) {
  zinstr(0x2A, dst, src1, src2, 0, nomask, EVEX_DT4 | EVEX_DT8 | EVEX_ENDS | EVEX_LIG | EVEX_M0F | EVEX_PF2 | EVEX_W0 | EVEX_W1);
}
void vcvtsi2ss(ZMMRegister dst, ZMMRegister src1, Register src2, RoundingMode er = noround) {
  zinstr(0x2A, dst, src1, src2, 0, nomask, EVEX_DT4 | EVEX_DT8 | EVEX_ENDS | EVEX_LIG | EVEX_M0F | EVEX_PF3 | EVEX_W0 | EVEX_W1 | evex_round(er));
}
void vcvtsi2ss(ZMMRegister dst, ZMMRegister src1, const Operand &src2) {
  zinstr(0x2A, dst, src1, src2, 0, nomask, EVEX_DT4 | EVEX_DT8 | EVEX_ENDS | EVEX_LIG | EVEX_M0F | EVEX_PF3 | EVEX_W0 | EVEX_W1);
}
void vcvtss2sd(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0x5A, dst, src1, src2, 0, mask, EVEX_DT4 | EVEX_ENDS | EVEX_LIG | EVEX_M0F | EVEX_PF3 | EVEX_SAE | EVEX_W0);
}
void vcvtss2sd(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x5A, dst, src1, src2, 0, mask, EVEX_DT4 | EVEX_ENDS | EVEX_LIG | EVEX_M0F | EVEX_PF3 | EVEX_SAE | EVEX_W0);
}
void vcvtss2si(Register dst, ZMMRegister src, RoundingMode er = noround) {
  zinstr(0x2D, dst, src, 0, nomask, EVEX_DT4 | EVEX_LIG | EVEX_M0F | EVEX_PF3 | EVEX_W0 | EVEX_W1 | evex_round(er));
}
void vcvtss2si(Register dst, const Operand &src) {
  zinstr(0x2D, dst, src, 0, nomask, EVEX_DT4 | EVEX_LIG | EVEX_M0F | EVEX_PF3 | EVEX_W0 | EVEX_W1);
}
void vcvtss2usi(Register dst, ZMMRegister src, RoundingMode er = noround) {
  zinstr(0x79, dst, src, 0, nomask, EVEX_DT4 | EVEX_LIG | EVEX_M0F | EVEX_PF3 | EVEX_W0 | EVEX_W1 | evex_round(er));
}
void vcvtss2usi(Register dst, const Operand &src) {
  zinstr(0x79, dst, src, 0, nomask, EVEX_DT4 | EVEX_LIG | EVEX_M0F | EVEX_PF3 | EVEX_W0 | EVEX_W1);
}
void vcvttpd2dq(ZMMRegister dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0xE6, dst, src, 0, mask, EVEX_BT8 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_SAE | EVEX_W1);
}
void vcvttpd2dq(ZMMRegister dst, const Operand &src, Mask mask = nomask) {
  zinstr(0xE6, dst, src, 0, mask, EVEX_BCST | EVEX_BT8 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W1);
}
void vcvttpd2qq(ZMMRegister dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x7A, dst, src, 0, mask, EVEX_BT8 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_SAE | EVEX_W1);
}
void vcvttpd2qq(ZMMRegister dst, const Operand &src, Mask mask = nomask) {
  zinstr(0x7A, dst, src, 0, mask, EVEX_BCST | EVEX_BT8 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W1);
}
void vcvttpd2udq(ZMMRegister dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x78, dst, src, 0, mask, EVEX_BT8 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_SAE | EVEX_W1);
}
void vcvttpd2udq(ZMMRegister dst, const Operand &src, Mask mask = nomask) {
  zinstr(0x78, dst, src, 0, mask, EVEX_BCST | EVEX_BT8 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_W1);
}
void vcvttps2dq(ZMMRegister dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x5B, dst, src, 0, mask, EVEX_BT4 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_PF3 | EVEX_SAE | EVEX_W0);
}
void vcvttps2dq(ZMMRegister dst, const Operand &src, Mask mask = nomask) {
  zinstr(0x5B, dst, src, 0, mask, EVEX_BCST | EVEX_BT4 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_PF3 | EVEX_W0);
}
void vcvttps2udq(ZMMRegister dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x78, dst, src, 0, mask, EVEX_BT4 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_SAE | EVEX_W0);
}
void vcvttps2udq(ZMMRegister dst, const Operand &src, Mask mask = nomask) {
  zinstr(0x78, dst, src, 0, mask, EVEX_BCST | EVEX_BT4 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_W0);
}
void vcvttsd2usi(Register dst, ZMMRegister src) {
  zinstr(0x78, dst, src, 0, nomask, EVEX_DT8 | EVEX_LIG | EVEX_M0F | EVEX_PF2 | EVEX_SAE | EVEX_W0 | EVEX_W1);
}
void vcvttsd2usi(Register dst, const Operand &src) {
  zinstr(0x78, dst, src, 0, nomask, EVEX_DT8 | EVEX_LIG | EVEX_M0F | EVEX_PF2 | EVEX_SAE | EVEX_W0 | EVEX_W1);
}
void vcvttss2usi(Register dst, ZMMRegister src) {
  zinstr(0x78, dst, src, 0, nomask, EVEX_DT4 | EVEX_LIG | EVEX_M0F | EVEX_PF3 | EVEX_SAE | EVEX_W0 | EVEX_W1);
}
void vcvttss2usi(Register dst, const Operand &src) {
  zinstr(0x78, dst, src, 0, nomask, EVEX_DT4 | EVEX_LIG | EVEX_M0F | EVEX_PF3 | EVEX_SAE | EVEX_W0 | EVEX_W1);
}
void vcvtudq2pd(ZMMRegister dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x7A, dst, src, 0, mask, EVEX_BT4 | EVEX_DT8 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_PF3 | EVEX_W0);
}
void vcvtudq2pd(ZMMRegister dst, const Operand &src, Mask mask = nomask) {
  zinstr(0x7A, dst, src, 0, mask, EVEX_BCST | EVEX_BT4 | EVEX_DT8 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_PF3 | EVEX_W0);
}
void vcvtudq2ps(ZMMRegister dst, ZMMRegister src, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0x7A, dst, src, 0, mask, EVEX_BT4 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_PF2 | EVEX_W0 | evex_round(er));
}
void vcvtudq2ps(ZMMRegister dst, const Operand &src, Mask mask = nomask) {
  zinstr(0x7A, dst, src, 0, mask, EVEX_BCST | EVEX_BT4 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_PF2 | EVEX_W0);
}
void vcvtusi2sd(ZMMRegister dst, ZMMRegister src1, Register src2, RoundingMode er = noround) {
  zinstr(0x7B, dst, src1, src2, 0, nomask, EVEX_DT4 | EVEX_DT8 | EVEX_ENDS | EVEX_LIG | EVEX_M0F | EVEX_PF2 | EVEX_W0 | EVEX_W1 | evex_round(er));
}
void vcvtusi2sd(ZMMRegister dst, ZMMRegister src1, const Operand &src2) {
  zinstr(0x7B, dst, src1, src2, 0, nomask, EVEX_DT4 | EVEX_DT8 | EVEX_ENDS | EVEX_LIG | EVEX_M0F | EVEX_PF2 | EVEX_W0 | EVEX_W1);
}
void vcvtusi2ss(ZMMRegister dst, ZMMRegister src1, Register src2, RoundingMode er = noround) {
  zinstr(0x7B, dst, src1, src2, 0, nomask, EVEX_DT4 | EVEX_DT8 | EVEX_ENDS | EVEX_LIG | EVEX_M0F | EVEX_PF3 | EVEX_W0 | EVEX_W1 | evex_round(er));
}
void vcvtusi2ss(ZMMRegister dst, ZMMRegister src1, const Operand &src2) {
  zinstr(0x7B, dst, src1, src2, 0, nomask, EVEX_DT4 | EVEX_DT8 | EVEX_ENDS | EVEX_LIG | EVEX_M0F | EVEX_PF3 | EVEX_W0 | EVEX_W1);
}
void vdivpd(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0x5E, dst, src1, src2, 0, mask, EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W1 | evex_round(er));
}
void vdivpd(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x5E, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W1);
}
void vdivps(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0x5E, dst, src1, src2, 0, mask, EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_W0 | evex_round(er));
}
void vdivps(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x5E, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_W0);
}
void vdivsd(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0x5E, dst, src1, src2, 0, mask, EVEX_DT8 | EVEX_ENDS | EVEX_LIG | EVEX_M0F | EVEX_PF2 | EVEX_W1 | evex_round(er));
}
void vdivsd(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x5E, dst, src1, src2, 0, mask, EVEX_DT8 | EVEX_ENDS | EVEX_LIG | EVEX_M0F | EVEX_PF2 | EVEX_W1);
}
void vdivss(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0x5E, dst, src1, src2, 0, mask, EVEX_DT4 | EVEX_ENDS | EVEX_LIG | EVEX_M0F | EVEX_PF3 | EVEX_W0 | evex_round(er));
}
void vdivss(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x5E, dst, src1, src2, 0, mask, EVEX_DT4 | EVEX_ENDS | EVEX_LIG | EVEX_M0F | EVEX_PF3 | EVEX_W0);
}
void vexpandpd(ZMMRegister dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x88, dst, src, 0, mask, EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vexpandpd(ZMMRegister dst, const Operand &src, Mask mask = nomask) {
  zinstr(0x88, dst, src, 0, mask, EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vexpandps(ZMMRegister dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x88, dst, src, 0, mask, EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vexpandps(ZMMRegister dst, const Operand &src, Mask mask = nomask) {
  zinstr(0x88, dst, src, 0, mask, EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vextractf32x4(ZMMRegister dst, ZMMRegister src, int8_t imm8, Mask mask = nomask) {
  zinstr(0x19, dst, src, imm8, mask, EVEX_IMM | EVEX_L256 | EVEX_L512 | EVEX_M0F3A | EVEX_P66 | EVEX_W0);
}
void vextractf32x4(const Operand &dst, ZMMRegister src, int8_t imm8, Mask mask = nomask) {
  zinstr(0x19, dst, src, imm8, mask, EVEX_IMM | EVEX_L256 | EVEX_L512 | EVEX_M0F3A | EVEX_P66 | EVEX_W0);
}
void vextractf64x4(ZMMRegister dst, ZMMRegister src, int8_t imm8, Mask mask = nomask) {
  zinstr(0x1B, dst, src, imm8, mask, EVEX_IMM | EVEX_L512 | EVEX_M0F3A | EVEX_P66 | EVEX_W1);
}
void vextractf64x4(const Operand &dst, ZMMRegister src, int8_t imm8, Mask mask = nomask) {
  zinstr(0x1B, dst, src, imm8, mask, EVEX_IMM | EVEX_L512 | EVEX_M0F3A | EVEX_P66 | EVEX_W1);
}
void vextracti32x4(ZMMRegister dst, ZMMRegister src, int8_t imm8, Mask mask = nomask) {
  zinstr(0x39, dst, src, imm8, mask, EVEX_IMM | EVEX_L256 | EVEX_L512 | EVEX_M0F3A | EVEX_P66 | EVEX_W0);
}
void vextracti32x4(const Operand &dst, ZMMRegister src, int8_t imm8, Mask mask = nomask) {
  zinstr(0x39, dst, src, imm8, mask, EVEX_IMM | EVEX_L256 | EVEX_L512 | EVEX_M0F3A | EVEX_P66 | EVEX_W0);
}
void vextracti64x4(ZMMRegister dst, ZMMRegister src, int8_t imm8, Mask mask = nomask) {
  zinstr(0x3B, dst, src, imm8, mask, EVEX_IMM | EVEX_L512 | EVEX_M0F3A | EVEX_P66 | EVEX_W1);
}
void vextracti64x4(const Operand &dst, ZMMRegister src, int8_t imm8, Mask mask = nomask) {
  zinstr(0x3B, dst, src, imm8, mask, EVEX_IMM | EVEX_L512 | EVEX_M0F3A | EVEX_P66 | EVEX_W1);
}
void vextractps(Register dst, ZMMRegister src, int8_t imm8) {
  zinstr(0x17, dst, src, imm8, nomask, EVEX_DT4 | EVEX_IMM | EVEX_L128 | EVEX_M0F3A | EVEX_P66 | EVEX_WIG);
}
void vextractps(const Operand &dst, ZMMRegister src, int8_t imm8) {
  zinstr(0x17, dst, src, imm8, nomask, EVEX_DT4 | EVEX_IMM | EVEX_L128 | EVEX_M0F3A | EVEX_P66 | EVEX_WIG);
}
void vfixupimmpd(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, int8_t imm8, Mask mask = nomask) {
  zinstr(0x54, dst, src1, src2, imm8, mask, EVEX_BT8 | EVEX_ENDS | EVEX_IMM | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F3A | EVEX_P66 | EVEX_SAE | EVEX_W1);
}
void vfixupimmpd(ZMMRegister dst, ZMMRegister src1, const Operand &src2, int8_t imm8, Mask mask = nomask) {
  zinstr(0x54, dst, src1, src2, imm8, mask, EVEX_BCST | EVEX_BT8 | EVEX_ENDS | EVEX_IMM | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F3A | EVEX_P66 | EVEX_W1);
}
void vfixupimmps(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, int8_t imm8, Mask mask = nomask) {
  zinstr(0x54, dst, src1, src2, imm8, mask, EVEX_BT4 | EVEX_ENDS | EVEX_IMM | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F3A | EVEX_P66 | EVEX_SAE | EVEX_W0);
}
void vfixupimmps(ZMMRegister dst, ZMMRegister src1, const Operand &src2, int8_t imm8, Mask mask = nomask) {
  zinstr(0x54, dst, src1, src2, imm8, mask, EVEX_BCST | EVEX_BT4 | EVEX_ENDS | EVEX_IMM | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F3A | EVEX_P66 | EVEX_W0);
}
void vfixupimmsd(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, int8_t imm8, Mask mask = nomask) {
  zinstr(0x55, dst, src1, src2, imm8, mask, EVEX_DT8 | EVEX_ENDS | EVEX_IMM | EVEX_LIG | EVEX_M0F3A | EVEX_P66 | EVEX_SAE | EVEX_W1);
}
void vfixupimmsd(ZMMRegister dst, ZMMRegister src1, const Operand &src2, int8_t imm8, Mask mask = nomask) {
  zinstr(0x55, dst, src1, src2, imm8, mask, EVEX_DT8 | EVEX_ENDS | EVEX_IMM | EVEX_LIG | EVEX_M0F3A | EVEX_P66 | EVEX_SAE | EVEX_W1);
}
void vfixupimmss(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, int8_t imm8, Mask mask = nomask) {
  zinstr(0x55, dst, src1, src2, imm8, mask, EVEX_DT4 | EVEX_ENDS | EVEX_IMM | EVEX_LIG | EVEX_M0F3A | EVEX_P66 | EVEX_SAE | EVEX_W0);
}
void vfixupimmss(ZMMRegister dst, ZMMRegister src1, const Operand &src2, int8_t imm8, Mask mask = nomask) {
  zinstr(0x55, dst, src1, src2, imm8, mask, EVEX_DT4 | EVEX_ENDS | EVEX_IMM | EVEX_LIG | EVEX_M0F3A | EVEX_P66 | EVEX_SAE | EVEX_W0);
}
void vfmadd132pd(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0x98, dst, src1, src2, 0, mask, EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1 | evex_round(er));
}
void vfmadd132pd(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x98, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vfmadd132ps(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0x98, dst, src1, src2, 0, mask, EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0 | evex_round(er));
}
void vfmadd132ps(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x98, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vfmadd132sd(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0x99, dst, src1, src2, 0, mask, EVEX_DT8 | EVEX_EDDS | EVEX_LIG | EVEX_M0F38 | EVEX_P66 | EVEX_W1 | evex_round(er));
}
void vfmadd132sd(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x99, dst, src1, src2, 0, mask, EVEX_DT8 | EVEX_EDDS | EVEX_LIG | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vfmadd132ss(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0x99, dst, src1, src2, 0, mask, EVEX_DT4 | EVEX_EDDS | EVEX_LIG | EVEX_M0F38 | EVEX_P66 | EVEX_W0 | evex_round(er));
}
void vfmadd132ss(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x99, dst, src1, src2, 0, mask, EVEX_DT4 | EVEX_EDDS | EVEX_LIG | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vfmadd213pd(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0xA8, dst, src1, src2, 0, mask, EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1 | evex_round(er));
}
void vfmadd213pd(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0xA8, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vfmadd213ps(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0xA8, dst, src1, src2, 0, mask, EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0 | evex_round(er));
}
void vfmadd213ps(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0xA8, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vfmadd213sd(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0xA9, dst, src1, src2, 0, mask, EVEX_DT8 | EVEX_EDDS | EVEX_LIG | EVEX_M0F38 | EVEX_P66 | EVEX_W1 | evex_round(er));
}
void vfmadd213sd(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0xA9, dst, src1, src2, 0, mask, EVEX_DT8 | EVEX_EDDS | EVEX_LIG | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vfmadd213ss(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0xA9, dst, src1, src2, 0, mask, EVEX_DT4 | EVEX_EDDS | EVEX_LIG | EVEX_M0F38 | EVEX_P66 | EVEX_W0 | evex_round(er));
}
void vfmadd213ss(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0xA9, dst, src1, src2, 0, mask, EVEX_DT4 | EVEX_EDDS | EVEX_LIG | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vfmadd231pd(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0xB8, dst, src1, src2, 0, mask, EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1 | evex_round(er));
}
void vfmadd231pd(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0xB8, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vfmadd231ps(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0xB8, dst, src1, src2, 0, mask, EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0 | evex_round(er));
}
void vfmadd231ps(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0xB8, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vfmadd231sd(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0xB9, dst, src1, src2, 0, mask, EVEX_DT8 | EVEX_EDDS | EVEX_LIG | EVEX_M0F38 | EVEX_P66 | EVEX_W1 | evex_round(er));
}
void vfmadd231sd(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0xB9, dst, src1, src2, 0, mask, EVEX_DT8 | EVEX_EDDS | EVEX_LIG | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vfmadd231ss(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0xB9, dst, src1, src2, 0, mask, EVEX_DT4 | EVEX_EDDS | EVEX_LIG | EVEX_M0F38 | EVEX_P66 | EVEX_W0 | evex_round(er));
}
void vfmadd231ss(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0xB9, dst, src1, src2, 0, mask, EVEX_DT4 | EVEX_EDDS | EVEX_LIG | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vfmaddsub132pd(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0x96, dst, src1, src2, 0, mask, EVEX_BT8 | EVEX_EDDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1 | evex_round(er));
}
void vfmaddsub132pd(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x96, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT8 | EVEX_EDDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vfmaddsub132ps(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0x96, dst, src1, src2, 0, mask, EVEX_BT4 | EVEX_EDDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0 | evex_round(er));
}
void vfmaddsub132ps(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x96, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT4 | EVEX_EDDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vfmaddsub213pd(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0xA6, dst, src1, src2, 0, mask, EVEX_BT8 | EVEX_EDDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1 | evex_round(er));
}
void vfmaddsub213pd(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0xA6, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT8 | EVEX_EDDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vfmaddsub213ps(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0xA6, dst, src1, src2, 0, mask, EVEX_BT4 | EVEX_EDDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0 | evex_round(er));
}
void vfmaddsub213ps(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0xA6, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT4 | EVEX_EDDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vfmaddsub231pd(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0xB6, dst, src1, src2, 0, mask, EVEX_BT8 | EVEX_EDDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1 | evex_round(er));
}
void vfmaddsub231pd(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0xB6, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT8 | EVEX_EDDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vfmaddsub231ps(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0xB6, dst, src1, src2, 0, mask, EVEX_BT4 | EVEX_EDDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0 | evex_round(er));
}
void vfmaddsub231ps(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0xB6, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT4 | EVEX_EDDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vfmsub132pd(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0x9A, dst, src1, src2, 0, mask, EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1 | evex_round(er));
}
void vfmsub132pd(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x9A, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vfmsub132ps(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0x9A, dst, src1, src2, 0, mask, EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0 | evex_round(er));
}
void vfmsub132ps(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x9A, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vfmsub132sd(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0x9B, dst, src1, src2, 0, mask, EVEX_DT8 | EVEX_EDDS | EVEX_LIG | EVEX_M0F38 | EVEX_P66 | EVEX_W1 | evex_round(er));
}
void vfmsub132sd(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x9B, dst, src1, src2, 0, mask, EVEX_DT8 | EVEX_EDDS | EVEX_LIG | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vfmsub132ss(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0x9B, dst, src1, src2, 0, mask, EVEX_DT4 | EVEX_EDDS | EVEX_LIG | EVEX_M0F38 | EVEX_P66 | EVEX_W0 | evex_round(er));
}
void vfmsub132ss(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x9B, dst, src1, src2, 0, mask, EVEX_DT4 | EVEX_EDDS | EVEX_LIG | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vfmsub213pd(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0xAA, dst, src1, src2, 0, mask, EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1 | evex_round(er));
}
void vfmsub213pd(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0xAA, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vfmsub213ps(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0xAA, dst, src1, src2, 0, mask, EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0 | evex_round(er));
}
void vfmsub213ps(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0xAA, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vfmsub213sd(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0xAB, dst, src1, src2, 0, mask, EVEX_DT8 | EVEX_EDDS | EVEX_LIG | EVEX_M0F38 | EVEX_P66 | EVEX_W1 | evex_round(er));
}
void vfmsub213sd(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0xAB, dst, src1, src2, 0, mask, EVEX_DT8 | EVEX_EDDS | EVEX_LIG | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vfmsub213ss(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0xAB, dst, src1, src2, 0, mask, EVEX_DT4 | EVEX_EDDS | EVEX_LIG | EVEX_M0F38 | EVEX_P66 | EVEX_W0 | evex_round(er));
}
void vfmsub213ss(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0xAB, dst, src1, src2, 0, mask, EVEX_DT4 | EVEX_EDDS | EVEX_LIG | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vfmsub231pd(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0xBA, dst, src1, src2, 0, mask, EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1 | evex_round(er));
}
void vfmsub231pd(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0xBA, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vfmsub231ps(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0xBA, dst, src1, src2, 0, mask, EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0 | evex_round(er));
}
void vfmsub231ps(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0xBA, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vfmsub231sd(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0xBB, dst, src1, src2, 0, mask, EVEX_DT8 | EVEX_EDDS | EVEX_LIG | EVEX_M0F38 | EVEX_P66 | EVEX_W1 | evex_round(er));
}
void vfmsub231sd(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0xBB, dst, src1, src2, 0, mask, EVEX_DT8 | EVEX_EDDS | EVEX_LIG | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vfmsub231ss(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0xBB, dst, src1, src2, 0, mask, EVEX_DT4 | EVEX_EDDS | EVEX_LIG | EVEX_M0F38 | EVEX_P66 | EVEX_W0 | evex_round(er));
}
void vfmsub231ss(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0xBB, dst, src1, src2, 0, mask, EVEX_DT4 | EVEX_EDDS | EVEX_LIG | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vfmsubadd132pd(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0x97, dst, src1, src2, 0, mask, EVEX_BT8 | EVEX_EDDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1 | evex_round(er));
}
void vfmsubadd132pd(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x97, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT8 | EVEX_EDDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vfmsubadd132ps(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0x97, dst, src1, src2, 0, mask, EVEX_BT4 | EVEX_EDDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0 | evex_round(er));
}
void vfmsubadd132ps(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x97, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT4 | EVEX_EDDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vfmsubadd213pd(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0xA7, dst, src1, src2, 0, mask, EVEX_BT8 | EVEX_EDDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1 | evex_round(er));
}
void vfmsubadd213pd(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0xA7, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT8 | EVEX_EDDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vfmsubadd213ps(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0xA7, dst, src1, src2, 0, mask, EVEX_BT4 | EVEX_EDDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0 | evex_round(er));
}
void vfmsubadd213ps(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0xA7, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT4 | EVEX_EDDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vfmsubadd231pd(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0xB7, dst, src1, src2, 0, mask, EVEX_BT8 | EVEX_EDDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1 | evex_round(er));
}
void vfmsubadd231pd(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0xB7, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT8 | EVEX_EDDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vfmsubadd231ps(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0xB7, dst, src1, src2, 0, mask, EVEX_BT4 | EVEX_EDDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0 | evex_round(er));
}
void vfmsubadd231ps(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0xB7, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT4 | EVEX_EDDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vfnmadd132pd(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0x9C, dst, src1, src2, 0, mask, EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1 | evex_round(er));
}
void vfnmadd132pd(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x9C, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vfnmadd132ps(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0x9C, dst, src1, src2, 0, mask, EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0 | evex_round(er));
}
void vfnmadd132ps(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x9C, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vfnmadd132sd(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0x9D, dst, src1, src2, 0, mask, EVEX_DT8 | EVEX_EDDS | EVEX_LIG | EVEX_M0F38 | EVEX_P66 | EVEX_W1 | evex_round(er));
}
void vfnmadd132sd(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x9D, dst, src1, src2, 0, mask, EVEX_DT8 | EVEX_EDDS | EVEX_LIG | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vfnmadd132ss(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0x9D, dst, src1, src2, 0, mask, EVEX_DT4 | EVEX_EDDS | EVEX_LIG | EVEX_M0F38 | EVEX_P66 | EVEX_W0 | evex_round(er));
}
void vfnmadd132ss(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x9D, dst, src1, src2, 0, mask, EVEX_DT4 | EVEX_EDDS | EVEX_LIG | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vfnmadd213pd(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0xAC, dst, src1, src2, 0, mask, EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1 | evex_round(er));
}
void vfnmadd213pd(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0xAC, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vfnmadd213ps(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0xAC, dst, src1, src2, 0, mask, EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0 | evex_round(er));
}
void vfnmadd213ps(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0xAC, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vfnmadd213sd(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0xAD, dst, src1, src2, 0, mask, EVEX_DT8 | EVEX_EDDS | EVEX_LIG | EVEX_M0F38 | EVEX_P66 | EVEX_W1 | evex_round(er));
}
void vfnmadd213sd(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0xAD, dst, src1, src2, 0, mask, EVEX_DT8 | EVEX_EDDS | EVEX_LIG | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vfnmadd213ss(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0xAD, dst, src1, src2, 0, mask, EVEX_DT4 | EVEX_EDDS | EVEX_LIG | EVEX_M0F38 | EVEX_P66 | EVEX_W0 | evex_round(er));
}
void vfnmadd213ss(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0xAD, dst, src1, src2, 0, mask, EVEX_DT4 | EVEX_EDDS | EVEX_LIG | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vfnmadd231pd(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0xBC, dst, src1, src2, 0, mask, EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1 | evex_round(er));
}
void vfnmadd231pd(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0xBC, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vfnmadd231ps(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0xBC, dst, src1, src2, 0, mask, EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0 | evex_round(er));
}
void vfnmadd231ps(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0xBC, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vfnmadd231sd(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0xBD, dst, src1, src2, 0, mask, EVEX_DT8 | EVEX_EDDS | EVEX_LIG | EVEX_M0F38 | EVEX_P66 | EVEX_W1 | evex_round(er));
}
void vfnmadd231sd(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0xBD, dst, src1, src2, 0, mask, EVEX_DT8 | EVEX_EDDS | EVEX_LIG | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vfnmadd231ss(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0xBD, dst, src1, src2, 0, mask, EVEX_DT4 | EVEX_EDDS | EVEX_LIG | EVEX_M0F38 | EVEX_P66 | EVEX_W0 | evex_round(er));
}
void vfnmadd231ss(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0xBD, dst, src1, src2, 0, mask, EVEX_DT4 | EVEX_EDDS | EVEX_LIG | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vfnmsub132pd(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0x9E, dst, src1, src2, 0, mask, EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1 | evex_round(er));
}
void vfnmsub132pd(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x9E, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vfnmsub132ps(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0x9E, dst, src1, src2, 0, mask, EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0 | evex_round(er));
}
void vfnmsub132ps(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x9E, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vfnmsub132sd(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0x9F, dst, src1, src2, 0, mask, EVEX_DT8 | EVEX_EDDS | EVEX_LIG | EVEX_M0F38 | EVEX_P66 | EVEX_W1 | evex_round(er));
}
void vfnmsub132sd(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x9F, dst, src1, src2, 0, mask, EVEX_DT8 | EVEX_EDDS | EVEX_LIG | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vfnmsub132ss(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0x9F, dst, src1, src2, 0, mask, EVEX_DT4 | EVEX_EDDS | EVEX_LIG | EVEX_M0F38 | EVEX_P66 | EVEX_W0 | evex_round(er));
}
void vfnmsub132ss(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x9F, dst, src1, src2, 0, mask, EVEX_DT4 | EVEX_EDDS | EVEX_LIG | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vfnmsub213pd(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0xAE, dst, src1, src2, 0, mask, EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1 | evex_round(er));
}
void vfnmsub213pd(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0xAE, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vfnmsub213ps(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0xAE, dst, src1, src2, 0, mask, EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0 | evex_round(er));
}
void vfnmsub213ps(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0xAE, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vfnmsub213sd(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0xAF, dst, src1, src2, 0, mask, EVEX_DT8 | EVEX_EDDS | EVEX_LIG | EVEX_M0F38 | EVEX_P66 | EVEX_W1 | evex_round(er));
}
void vfnmsub213sd(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0xAF, dst, src1, src2, 0, mask, EVEX_DT8 | EVEX_EDDS | EVEX_LIG | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vfnmsub213ss(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0xAF, dst, src1, src2, 0, mask, EVEX_DT4 | EVEX_EDDS | EVEX_LIG | EVEX_M0F38 | EVEX_P66 | EVEX_W0 | evex_round(er));
}
void vfnmsub213ss(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0xAF, dst, src1, src2, 0, mask, EVEX_DT4 | EVEX_EDDS | EVEX_LIG | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vfnmsub231pd(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0xBE, dst, src1, src2, 0, mask, EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1 | evex_round(er));
}
void vfnmsub231pd(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0xBE, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vfnmsub231ps(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0xBE, dst, src1, src2, 0, mask, EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0 | evex_round(er));
}
void vfnmsub231ps(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0xBE, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vfnmsub231sd(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0xBF, dst, src1, src2, 0, mask, EVEX_DT8 | EVEX_EDDS | EVEX_LIG | EVEX_M0F38 | EVEX_P66 | EVEX_W1 | evex_round(er));
}
void vfnmsub231sd(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0xBF, dst, src1, src2, 0, mask, EVEX_DT8 | EVEX_EDDS | EVEX_LIG | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vfnmsub231ss(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0xBF, dst, src1, src2, 0, mask, EVEX_DT4 | EVEX_EDDS | EVEX_LIG | EVEX_M0F38 | EVEX_P66 | EVEX_W0 | evex_round(er));
}
void vfnmsub231ss(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0xBF, dst, src1, src2, 0, mask, EVEX_DT4 | EVEX_EDDS | EVEX_LIG | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vgetexppd(ZMMRegister dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x42, dst, src, 0, mask, EVEX_BT8 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_SAE | EVEX_W1);
}
void vgetexppd(ZMMRegister dst, const Operand &src, Mask mask = nomask) {
  zinstr(0x42, dst, src, 0, mask, EVEX_BCST | EVEX_BT8 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vgetexpps(ZMMRegister dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x42, dst, src, 0, mask, EVEX_BT4 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_SAE | EVEX_W0);
}
void vgetexpps(ZMMRegister dst, const Operand &src, Mask mask = nomask) {
  zinstr(0x42, dst, src, 0, mask, EVEX_BCST | EVEX_BT4 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vgetexpsd(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0x43, dst, src1, src2, 0, mask, EVEX_DT8 | EVEX_ENDS | EVEX_LIG | EVEX_M0F38 | EVEX_P66 | EVEX_SAE | EVEX_W1);
}
void vgetexpsd(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x43, dst, src1, src2, 0, mask, EVEX_DT8 | EVEX_ENDS | EVEX_LIG | EVEX_M0F38 | EVEX_P66 | EVEX_SAE | EVEX_W1);
}
void vgetexpss(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0x43, dst, src1, src2, 0, mask, EVEX_DT4 | EVEX_ENDS | EVEX_LIG | EVEX_M0F38 | EVEX_P66 | EVEX_SAE | EVEX_W0);
}
void vgetexpss(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x43, dst, src1, src2, 0, mask, EVEX_DT4 | EVEX_ENDS | EVEX_LIG | EVEX_M0F38 | EVEX_P66 | EVEX_SAE | EVEX_W0);
}
void vgetmantpd(ZMMRegister dst, ZMMRegister src, int8_t imm8, Mask mask = nomask) {
  zinstr(0x26, dst, src, imm8, mask, EVEX_BT8 | EVEX_IMM | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F3A | EVEX_P66 | EVEX_SAE | EVEX_W1);
}
void vgetmantpd(ZMMRegister dst, const Operand &src, int8_t imm8, Mask mask = nomask) {
  zinstr(0x26, dst, src, imm8, mask, EVEX_BCST | EVEX_BT8 | EVEX_IMM | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F3A | EVEX_P66 | EVEX_W1);
}
void vgetmantps(ZMMRegister dst, ZMMRegister src, int8_t imm8, Mask mask = nomask) {
  zinstr(0x26, dst, src, imm8, mask, EVEX_BT4 | EVEX_IMM | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F3A | EVEX_P66 | EVEX_SAE | EVEX_W0);
}
void vgetmantps(ZMMRegister dst, const Operand &src, int8_t imm8, Mask mask = nomask) {
  zinstr(0x26, dst, src, imm8, mask, EVEX_BCST | EVEX_BT4 | EVEX_IMM | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F3A | EVEX_P66 | EVEX_W0);
}
void vgetmantsd(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, int8_t imm8, Mask mask = nomask) {
  zinstr(0x27, dst, src1, src2, imm8, mask, EVEX_DT8 | EVEX_ENDS | EVEX_IMM | EVEX_LIG | EVEX_M0F3A | EVEX_P66 | EVEX_SAE | EVEX_W1);
}
void vgetmantsd(ZMMRegister dst, ZMMRegister src1, const Operand &src2, int8_t imm8, Mask mask = nomask) {
  zinstr(0x27, dst, src1, src2, imm8, mask, EVEX_DT8 | EVEX_ENDS | EVEX_IMM | EVEX_LIG | EVEX_M0F3A | EVEX_P66 | EVEX_SAE | EVEX_W1);
}
void vgetmantss(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, int8_t imm8, Mask mask = nomask) {
  zinstr(0x27, dst, src1, src2, imm8, mask, EVEX_DT4 | EVEX_ENDS | EVEX_IMM | EVEX_LIG | EVEX_M0F3A | EVEX_P66 | EVEX_SAE | EVEX_W0);
}
void vgetmantss(ZMMRegister dst, ZMMRegister src1, const Operand &src2, int8_t imm8, Mask mask = nomask) {
  zinstr(0x27, dst, src1, src2, imm8, mask, EVEX_DT4 | EVEX_ENDS | EVEX_IMM | EVEX_LIG | EVEX_M0F3A | EVEX_P66 | EVEX_SAE | EVEX_W0);
}
void vinsertf32x4(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, int8_t imm8, Mask mask = nomask) {
  zinstr(0x18, dst, src1, src2, imm8, mask, EVEX_ENDS | EVEX_IMM | EVEX_L256 | EVEX_L512 | EVEX_M0F3A | EVEX_P66 | EVEX_W0);
}
void vinsertf32x4(ZMMRegister dst, ZMMRegister src1, const Operand &src2, int8_t imm8, Mask mask = nomask) {
  zinstr(0x18, dst, src1, src2, imm8, mask, EVEX_ENDS | EVEX_IMM | EVEX_L256 | EVEX_L512 | EVEX_M0F3A | EVEX_P66 | EVEX_W0);
}
void vinsertf64x4(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, int8_t imm8, Mask mask = nomask) {
  zinstr(0x1A, dst, src1, src2, imm8, mask, EVEX_ENDS | EVEX_IMM | EVEX_L512 | EVEX_M0F3A | EVEX_P66 | EVEX_W1);
}
void vinsertf64x4(ZMMRegister dst, ZMMRegister src1, const Operand &src2, int8_t imm8, Mask mask = nomask) {
  zinstr(0x1A, dst, src1, src2, imm8, mask, EVEX_ENDS | EVEX_IMM | EVEX_L512 | EVEX_M0F3A | EVEX_P66 | EVEX_W1);
}
void vinserti32x4(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, int8_t imm8, Mask mask = nomask) {
  zinstr(0x38, dst, src1, src2, imm8, mask, EVEX_ENDS | EVEX_IMM | EVEX_L256 | EVEX_L512 | EVEX_M0F3A | EVEX_P66 | EVEX_W0);
}
void vinserti32x4(ZMMRegister dst, ZMMRegister src1, const Operand &src2, int8_t imm8, Mask mask = nomask) {
  zinstr(0x38, dst, src1, src2, imm8, mask, EVEX_ENDS | EVEX_IMM | EVEX_L256 | EVEX_L512 | EVEX_M0F3A | EVEX_P66 | EVEX_W0);
}
void vinserti64x4(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, int8_t imm8, Mask mask = nomask) {
  zinstr(0x3A, dst, src1, src2, imm8, mask, EVEX_ENDS | EVEX_IMM | EVEX_L512 | EVEX_M0F3A | EVEX_P66 | EVEX_W1);
}
void vinserti64x4(ZMMRegister dst, ZMMRegister src1, const Operand &src2, int8_t imm8, Mask mask = nomask) {
  zinstr(0x3A, dst, src1, src2, imm8, mask, EVEX_ENDS | EVEX_IMM | EVEX_L512 | EVEX_M0F3A | EVEX_P66 | EVEX_W1);
}
void vinsertps(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, int8_t imm8) {
  zinstr(0x21, dst, src1, src2, imm8, nomask, EVEX_DT4 | EVEX_ENDS | EVEX_IMM | EVEX_L128 | EVEX_M0F3A | EVEX_P66 | EVEX_W0);
}
void vinsertps(ZMMRegister dst, ZMMRegister src1, const Operand &src2, int8_t imm8) {
  zinstr(0x21, dst, src1, src2, imm8, nomask, EVEX_DT4 | EVEX_ENDS | EVEX_IMM | EVEX_L128 | EVEX_M0F3A | EVEX_P66 | EVEX_W0);
}
void vmaxpd(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0x5F, dst, src1, src2, 0, mask, EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_SAE | EVEX_W1);
}
void vmaxpd(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x5F, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W1);
}
void vmaxps(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0x5F, dst, src1, src2, 0, mask, EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_SAE | EVEX_W0);
}
void vmaxps(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x5F, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_W0);
}
void vmaxsd(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0x5F, dst, src1, src2, 0, mask, EVEX_DT8 | EVEX_ENDS | EVEX_LIG | EVEX_M0F | EVEX_PF2 | EVEX_SAE | EVEX_W1);
}
void vmaxsd(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x5F, dst, src1, src2, 0, mask, EVEX_DT8 | EVEX_ENDS | EVEX_LIG | EVEX_M0F | EVEX_PF2 | EVEX_SAE | EVEX_W1);
}
void vmaxss(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0x5F, dst, src1, src2, 0, mask, EVEX_DT4 | EVEX_ENDS | EVEX_LIG | EVEX_M0F | EVEX_PF3 | EVEX_SAE | EVEX_W0);
}
void vmaxss(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x5F, dst, src1, src2, 0, mask, EVEX_DT4 | EVEX_ENDS | EVEX_LIG | EVEX_M0F | EVEX_PF3 | EVEX_SAE | EVEX_W0);
}
void vminpd(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0x5D, dst, src1, src2, 0, mask, EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_SAE | EVEX_W1);
}
void vminpd(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x5D, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W1);
}
void vminps(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0x5D, dst, src1, src2, 0, mask, EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_SAE | EVEX_W0);
}
void vminps(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x5D, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_W0);
}
void vminsd(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0x5D, dst, src1, src2, 0, mask, EVEX_DT8 | EVEX_ENDS | EVEX_LIG | EVEX_M0F | EVEX_PF2 | EVEX_SAE | EVEX_W1);
}
void vminsd(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x5D, dst, src1, src2, 0, mask, EVEX_DT8 | EVEX_ENDS | EVEX_LIG | EVEX_M0F | EVEX_PF2 | EVEX_SAE | EVEX_W1);
}
void vminss(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0x5D, dst, src1, src2, 0, mask, EVEX_DT4 | EVEX_ENDS | EVEX_LIG | EVEX_M0F | EVEX_PF3 | EVEX_SAE | EVEX_W0);
}
void vminss(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x5D, dst, src1, src2, 0, mask, EVEX_DT4 | EVEX_ENDS | EVEX_LIG | EVEX_M0F | EVEX_PF3 | EVEX_SAE | EVEX_W0);
}
void vmovapd(ZMMRegister dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x28, dst, src, 0, mask, EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W1);
}
void vmovapd(ZMMRegister dst, const Operand &src, Mask mask = nomask) {
  zinstr(0x28, dst, src, 0, mask, EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W1);
}
void vmovapd(const Operand &dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x29, dst, src, 0, mask, EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W1);
}
void vmovaps(ZMMRegister dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x28, dst, src, 0, mask, EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_W0);
}
void vmovaps(ZMMRegister dst, const Operand &src, Mask mask = nomask) {
  zinstr(0x28, dst, src, 0, mask, EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_W0);
}
void vmovaps(const Operand &dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x29, dst, src, 0, mask, EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_W0);
}
void vmovd(ZMMRegister dst, Register src) {
  zinstr(0x6E, dst, src, 0, nomask, EVEX_DT4 | EVEX_L128 | EVEX_M0F | EVEX_P66 | EVEX_W0);
}
void vmovd(Register dst, ZMMRegister src) {
  zinstr(0x7E, dst, src, 0, nomask, EVEX_DT4 | EVEX_L128 | EVEX_M0F | EVEX_P66 | EVEX_W0);
}
void vmovd(ZMMRegister dst, const Operand &src) {
  zinstr(0x6E, dst, src, 0, nomask, EVEX_DT4 | EVEX_L128 | EVEX_M0F | EVEX_P66 | EVEX_W0);
}
void vmovd(const Operand &dst, ZMMRegister src) {
  zinstr(0x7E, dst, src, 0, nomask, EVEX_DT4 | EVEX_L128 | EVEX_M0F | EVEX_P66 | EVEX_W0);
}
void vmovddup(ZMMRegister dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x12, dst, src, 0, mask, EVEX_DT8 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_PF2 | EVEX_W1);
}
void vmovddup(ZMMRegister dst, const Operand &src, Mask mask = nomask) {
  zinstr(0x12, dst, src, 0, mask, EVEX_DT8 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_PF2 | EVEX_W1);
}
void vmovdqa32(ZMMRegister dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x6F, dst, src, 0, mask, EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W0);
}
void vmovdqa32(ZMMRegister dst, const Operand &src, Mask mask = nomask) {
  zinstr(0x6F, dst, src, 0, mask, EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W0);
}
void vmovdqa32(const Operand &dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x7F, dst, src, 0, mask, EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W0);
}
void vmovdqa64(ZMMRegister dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x6F, dst, src, 0, mask, EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W1);
}
void vmovdqa64(ZMMRegister dst, const Operand &src, Mask mask = nomask) {
  zinstr(0x6F, dst, src, 0, mask, EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W1);
}
void vmovdqa64(const Operand &dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x7F, dst, src, 0, mask, EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W1);
}
void vmovdqu32(ZMMRegister dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x6F, dst, src, 0, mask, EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_PF3 | EVEX_W0);
}
void vmovdqu32(ZMMRegister dst, const Operand &src, Mask mask = nomask) {
  zinstr(0x6F, dst, src, 0, mask, EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_PF3 | EVEX_W0);
}
void vmovdqu32(const Operand &dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x7F, dst, src, 0, mask, EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_PF3 | EVEX_W0);
}
void vmovdqu64(ZMMRegister dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x6F, dst, src, 0, mask, EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_PF3 | EVEX_W1);
}
void vmovdqu64(ZMMRegister dst, const Operand &src, Mask mask = nomask) {
  zinstr(0x6F, dst, src, 0, mask, EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_PF3 | EVEX_W1);
}
void vmovdqu64(const Operand &dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x7F, dst, src, 0, mask, EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_PF3 | EVEX_W1);
}
void vmovhlps(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2) {
  zinstr(0x12, dst, src1, src2, 0, nomask, EVEX_ENDS | EVEX_L128 | EVEX_M0F | EVEX_W0);
}
void vmovhpd(ZMMRegister dst, ZMMRegister src1, const Operand &src2) {
  zinstr(0x16, dst, src1, src2, 0, nomask, EVEX_DT8 | EVEX_ENDS | EVEX_L128 | EVEX_M0F | EVEX_P66 | EVEX_W1);
}
void vmovhpd(const Operand &dst, ZMMRegister src) {
  zinstr(0x17, dst, src, 0, nomask, EVEX_DT8 | EVEX_L128 | EVEX_M0F | EVEX_P66 | EVEX_W1);
}
void vmovhps(ZMMRegister dst, ZMMRegister src1, const Operand &src2) {
  zinstr(0x16, dst, src1, src2, 0, nomask, EVEX_DT8 | EVEX_ENDS | EVEX_L128 | EVEX_M0F | EVEX_W0);
}
void vmovhps(const Operand &dst, ZMMRegister src) {
  zinstr(0x17, dst, src, 0, nomask, EVEX_DT8 | EVEX_L128 | EVEX_M0F | EVEX_W0);
}
void vmovlhps(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2) {
  zinstr(0x16, dst, src1, src2, 0, nomask, EVEX_ENDS | EVEX_L128 | EVEX_M0F | EVEX_W0);
}
void vmovlpd(ZMMRegister dst, ZMMRegister src1, const Operand &src2) {
  zinstr(0x12, dst, src1, src2, 0, nomask, EVEX_DT8 | EVEX_ENDS | EVEX_L128 | EVEX_M0F | EVEX_P66 | EVEX_W1);
}
void vmovlpd(const Operand &dst, ZMMRegister src) {
  zinstr(0x13, dst, src, 0, nomask, EVEX_DT8 | EVEX_L128 | EVEX_M0F | EVEX_P66 | EVEX_W1);
}
void vmovlps(ZMMRegister dst, ZMMRegister src1, const Operand &src2) {
  zinstr(0x12, dst, src1, src2, 0, nomask, EVEX_DT8 | EVEX_ENDS | EVEX_L128 | EVEX_M0F | EVEX_W0);
}
void vmovlps(const Operand &dst, ZMMRegister src) {
  zinstr(0x13, dst, src, 0, nomask, EVEX_DT8 | EVEX_L128 | EVEX_M0F | EVEX_W0);
}
void vmovntdq(const Operand &dst, ZMMRegister src) {
  zinstr(0xE7, dst, src, 0, nomask, EVEX_DT16 | EVEX_DT32 | EVEX_DT64 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W0);
}
void vmovntdqa(ZMMRegister dst, const Operand &src) {
  zinstr(0x2A, dst, src, 0, nomask, EVEX_DT16 | EVEX_DT32 | EVEX_DT64 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vmovntpd(const Operand &dst, ZMMRegister src) {
  zinstr(0x2B, dst, src, 0, nomask, EVEX_DT16 | EVEX_DT32 | EVEX_DT64 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W1);
}
void vmovntps(const Operand &dst, ZMMRegister src) {
  zinstr(0x2B, dst, src, 0, nomask, EVEX_DT16 | EVEX_DT32 | EVEX_DT64 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_W0);
}
void vmovq(ZMMRegister dst, Register src) {
  zinstr(0x6E, dst, src, 0, nomask, EVEX_DT8 | EVEX_L128 | EVEX_M0F | EVEX_P66 | EVEX_W1);
}
void vmovq(Register dst, ZMMRegister src) {
  zinstr(0x7E, dst, src, 0, nomask, EVEX_DT8 | EVEX_L128 | EVEX_M0F | EVEX_P66 | EVEX_W1);
}
void vmovq(ZMMRegister dst, ZMMRegister src) {
  zinstr(0x7E, dst, src, 0, nomask, EVEX_DT8 | EVEX_L128 | EVEX_M0F | EVEX_PF3 | EVEX_W1);
}
void vmovq(ZMMRegister dst, const Operand &src) {
  zinstr(0x6E, dst, src, 0, nomask, EVEX_DT8 | EVEX_L128 | EVEX_M0F | EVEX_P66 | EVEX_W1);
}
void vmovq(const Operand &dst, ZMMRegister src) {
  zinstr(0x7E, dst, src, 0, nomask, EVEX_DT8 | EVEX_L128 | EVEX_M0F | EVEX_P66 | EVEX_W1);
}
void vmovsd(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0x10, dst, src1, src2, 0, mask, EVEX_ENDS | EVEX_LIG | EVEX_M0F | EVEX_PF2 | EVEX_W1);
}
void vmovsd(ZMMRegister dst, const Operand &src, Mask mask = nomask) {
  zinstr(0x10, dst, src, 0, mask, EVEX_DT8 | EVEX_LIG | EVEX_M0F | EVEX_PF2 | EVEX_W1);
}
void vmovsd(const Operand &dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x11, dst, src, 0, mask, EVEX_DT8 | EVEX_LIG | EVEX_M0F | EVEX_PF2 | EVEX_W1);
}
void vmovshdup(ZMMRegister dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x16, dst, src, 0, mask, EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_PF3 | EVEX_W0);
}
void vmovshdup(ZMMRegister dst, const Operand &src, Mask mask = nomask) {
  zinstr(0x16, dst, src, 0, mask, EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_PF3 | EVEX_W0);
}
void vmovsldup(ZMMRegister dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x12, dst, src, 0, mask, EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_PF3 | EVEX_W0);
}
void vmovsldup(ZMMRegister dst, const Operand &src, Mask mask = nomask) {
  zinstr(0x12, dst, src, 0, mask, EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_PF3 | EVEX_W0);
}
void vmovss(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0x10, dst, src1, src2, 0, mask, EVEX_ENDS | EVEX_LIG | EVEX_M0F | EVEX_PF3 | EVEX_W0);
}
void vmovss(ZMMRegister dst, const Operand &src, Mask mask = nomask) {
  zinstr(0x10, dst, src, 0, mask, EVEX_DT4 | EVEX_LIG | EVEX_M0F | EVEX_PF3 | EVEX_W0);
}
void vmovss(const Operand &dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x11, dst, src, 0, mask, EVEX_DT4 | EVEX_LIG | EVEX_M0F | EVEX_PF3 | EVEX_W0);
}
void vmovupd(ZMMRegister dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x10, dst, src, 0, mask, EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W1);
}
void vmovupd(ZMMRegister dst, const Operand &src, Mask mask = nomask) {
  zinstr(0x10, dst, src, 0, mask, EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W1);
}
void vmovupd(const Operand &dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x11, dst, src, 0, mask, EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W1);
}
void vmovups(ZMMRegister dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x10, dst, src, 0, mask, EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_W0);
}
void vmovups(ZMMRegister dst, const Operand &src, Mask mask = nomask) {
  zinstr(0x10, dst, src, 0, mask, EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_W0);
}
void vmovups(const Operand &dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x11, dst, src, 0, mask, EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_W0);
}
void vmulpd(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0x59, dst, src1, src2, 0, mask, EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W1 | evex_round(er));
}
void vmulpd(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x59, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W1);
}
void vmulps(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0x59, dst, src1, src2, 0, mask, EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_W0 | evex_round(er));
}
void vmulps(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x59, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_W0);
}
void vmulsd(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0x59, dst, src1, src2, 0, mask, EVEX_DT8 | EVEX_ENDS | EVEX_LIG | EVEX_M0F | EVEX_PF2 | EVEX_W1 | evex_round(er));
}
void vmulsd(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x59, dst, src1, src2, 0, mask, EVEX_DT8 | EVEX_ENDS | EVEX_LIG | EVEX_M0F | EVEX_PF2 | EVEX_W1);
}
void vmulss(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0x59, dst, src1, src2, 0, mask, EVEX_DT4 | EVEX_ENDS | EVEX_LIG | EVEX_M0F | EVEX_PF3 | EVEX_W0 | evex_round(er));
}
void vmulss(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x59, dst, src1, src2, 0, mask, EVEX_DT4 | EVEX_ENDS | EVEX_LIG | EVEX_M0F | EVEX_PF3 | EVEX_W0);
}
void vorpd(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0x56, dst, src1, src2, 0, mask, EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W1);
}
void vorpd(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x56, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W1);
}
void vorps(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0x56, dst, src1, src2, 0, mask, EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_W0);
}
void vorps(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x56, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_W0);
}
void vpaddb(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0xFC, dst, src1, src2, 0, mask, EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_WIG);
}
void vpaddb(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0xFC, dst, src1, src2, 0, mask, EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_WIG);
}
void vpaddd(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0xFE, dst, src1, src2, 0, mask, EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W0);
}
void vpaddd(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0xFE, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W0);
}
void vpaddq(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0xD4, dst, src1, src2, 0, mask, EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W1);
}
void vpaddq(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0xD4, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W1);
}
void vpaddw(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0xFD, dst, src1, src2, 0, mask, EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_WIG);
}
void vpaddw(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0xFD, dst, src1, src2, 0, mask, EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_WIG);
}
void vpandd(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0xDB, dst, src1, src2, 0, mask, EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W0);
}
void vpandd(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0xDB, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W0);
}
void vpandnd(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0xDF, dst, src1, src2, 0, mask, EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W0);
}
void vpandnd(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0xDF, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W0);
}
void vpandnq(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0xDF, dst, src1, src2, 0, mask, EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W1);
}
void vpandnq(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0xDF, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W1);
}
void vpandq(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0xDB, dst, src1, src2, 0, mask, EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W1);
}
void vpandq(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0xDB, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W1);
}
void vpblendmd(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0x64, dst, src1, src2, 0, mask, EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vpblendmd(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x64, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vpblendmq(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0x64, dst, src1, src2, 0, mask, EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vpblendmq(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x64, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vpbroadcastd(ZMMRegister dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x58, dst, src, 0, mask, EVEX_DT4 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vpbroadcastd(ZMMRegister dst, Register src, Mask mask = nomask) {
  zinstr(0x7C, dst, src, 0, mask, EVEX_DT4 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vpbroadcastd(ZMMRegister dst, const Operand &src, Mask mask = nomask) {
  zinstr(0x58, dst, src, 0, mask, EVEX_DT4 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vpbroadcastq(ZMMRegister dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x59, dst, src, 0, mask, EVEX_DT8 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vpbroadcastq(ZMMRegister dst, Register src, Mask mask = nomask) {
  zinstr(0x7C, dst, src, 0, mask, EVEX_DT8 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vpbroadcastq(ZMMRegister dst, const Operand &src, Mask mask = nomask) {
  zinstr(0x59, dst, src, 0, mask, EVEX_DT8 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vpcmpd(OpmaskRegister dst, ZMMRegister src1, ZMMRegister src2, int8_t imm8, Mask mask = nomask) {
  zinstr(0x1F, dst, src1, src2, imm8, mask, EVEX_BT4 | EVEX_ENDS | EVEX_IMM | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F3A | EVEX_P66 | EVEX_W0);
}
void vpcmpd(OpmaskRegister dst, ZMMRegister src1, const Operand &src2, int8_t imm8, Mask mask = nomask) {
  zinstr(0x1F, dst, src1, src2, imm8, mask, EVEX_BCST | EVEX_BT4 | EVEX_ENDS | EVEX_IMM | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F3A | EVEX_P66 | EVEX_W0);
}
void vpcmpeqb(OpmaskRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0x74, dst, src1, src2, 0, mask, EVEX_ENDS | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_WIG);
}
void vpcmpeqb(OpmaskRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x74, dst, src1, src2, 0, mask, EVEX_ENDS | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_WIG);
}
void vpcmpeqd(OpmaskRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0x76, dst, src1, src2, 0, mask, EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W0);
}
void vpcmpeqd(OpmaskRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x76, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W0);
}
void vpcmpeqq(OpmaskRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0x29, dst, src1, src2, 0, mask, EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vpcmpeqq(OpmaskRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x29, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vpcmpeqw(OpmaskRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0x75, dst, src1, src2, 0, mask, EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_WIG);
}
void vpcmpeqw(OpmaskRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x75, dst, src1, src2, 0, mask, EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_WIG);
}
void vpcmpgtd(OpmaskRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0x66, dst, src1, src2, 0, mask, EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W0);
}
void vpcmpgtd(OpmaskRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x66, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W0);
}
void vpcmpgtq(OpmaskRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0x37, dst, src1, src2, 0, mask, EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vpcmpgtq(OpmaskRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x37, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vpcmpq(OpmaskRegister dst, ZMMRegister src1, ZMMRegister src2, int8_t imm8, Mask mask = nomask) {
  zinstr(0x1F, dst, src1, src2, imm8, mask, EVEX_BT8 | EVEX_ENDS | EVEX_IMM | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F3A | EVEX_P66 | EVEX_W1);
}
void vpcmpq(OpmaskRegister dst, ZMMRegister src1, const Operand &src2, int8_t imm8, Mask mask = nomask) {
  zinstr(0x1F, dst, src1, src2, imm8, mask, EVEX_BCST | EVEX_BT8 | EVEX_ENDS | EVEX_IMM | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F3A | EVEX_P66 | EVEX_W1);
}
void vpcmpud(OpmaskRegister dst, ZMMRegister src1, ZMMRegister src2, int8_t imm8, Mask mask = nomask) {
  zinstr(0x1E, dst, src1, src2, imm8, mask, EVEX_BT4 | EVEX_ENDS | EVEX_IMM | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F3A | EVEX_P66 | EVEX_W0);
}
void vpcmpud(OpmaskRegister dst, ZMMRegister src1, const Operand &src2, int8_t imm8, Mask mask = nomask) {
  zinstr(0x1E, dst, src1, src2, imm8, mask, EVEX_BCST | EVEX_BT4 | EVEX_ENDS | EVEX_IMM | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F3A | EVEX_P66 | EVEX_W0);
}
void vpcmpuq(OpmaskRegister dst, ZMMRegister src1, ZMMRegister src2, int8_t imm8, Mask mask = nomask) {
  zinstr(0x1E, dst, src1, src2, imm8, mask, EVEX_BT8 | EVEX_ENDS | EVEX_IMM | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F3A | EVEX_P66 | EVEX_W1);
}
void vpcmpuq(OpmaskRegister dst, ZMMRegister src1, const Operand &src2, int8_t imm8, Mask mask = nomask) {
  zinstr(0x1E, dst, src1, src2, imm8, mask, EVEX_BCST | EVEX_BT8 | EVEX_ENDS | EVEX_IMM | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F3A | EVEX_P66 | EVEX_W1);
}
void vpcompressd(ZMMRegister dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x8B, dst, src, 0, mask, EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vpcompressd(const Operand &dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x8B, dst, src, 0, mask, EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vpcompressq(ZMMRegister dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x8B, dst, src, 0, mask, EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vpcompressq(const Operand &dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x8B, dst, src, 0, mask, EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vpermd(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0x36, dst, src1, src2, 0, mask, EVEX_BT4 | EVEX_ENDS | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vpermd(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x36, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT4 | EVEX_ENDS | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vpermi2d(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0x76, dst, src1, src2, 0, mask, EVEX_BT4 | EVEX_EDDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vpermi2d(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x76, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT4 | EVEX_EDDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vpermi2pd(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0x77, dst, src1, src2, 0, mask, EVEX_BT8 | EVEX_EDDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vpermi2pd(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x77, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT8 | EVEX_EDDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vpermi2ps(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0x77, dst, src1, src2, 0, mask, EVEX_BT4 | EVEX_EDDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vpermi2ps(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x77, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT4 | EVEX_EDDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vpermi2q(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0x76, dst, src1, src2, 0, mask, EVEX_BT8 | EVEX_EDDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vpermi2q(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x76, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT8 | EVEX_EDDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vpermilpd(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0x0D, dst, src1, src2, 0, mask, EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vpermilpd(ZMMRegister dst, ZMMRegister src, int8_t imm8, Mask mask = nomask) {
  zinstr(0x05, dst, src, imm8, mask, EVEX_BT8 | EVEX_IMM | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F3A | EVEX_P66 | EVEX_W1);
}
void vpermilpd(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x0D, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vpermilpd(ZMMRegister dst, const Operand &src, int8_t imm8, Mask mask = nomask) {
  zinstr(0x05, dst, src, imm8, mask, EVEX_BCST | EVEX_BT8 | EVEX_IMM | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F3A | EVEX_P66 | EVEX_W1);
}
void vpermilps(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0x0C, dst, src1, src2, 0, mask, EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vpermilps(ZMMRegister dst, ZMMRegister src, int8_t imm8, Mask mask = nomask) {
  zinstr(0x04, dst, src, imm8, mask, EVEX_BT4 | EVEX_IMM | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F3A | EVEX_P66 | EVEX_W0);
}
void vpermilps(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x0C, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vpermilps(ZMMRegister dst, const Operand &src, int8_t imm8, Mask mask = nomask) {
  zinstr(0x04, dst, src, imm8, mask, EVEX_BCST | EVEX_BT4 | EVEX_IMM | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F3A | EVEX_P66 | EVEX_W0);
}
void vpermpd(ZMMRegister dst, ZMMRegister src, int8_t imm8, Mask mask = nomask) {
  zinstr(0x01, dst, src, imm8, mask, EVEX_BT8 | EVEX_IMM | EVEX_L256 | EVEX_L512 | EVEX_M0F3A | EVEX_P66 | EVEX_W1);
}
void vpermpd(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0x16, dst, src1, src2, 0, mask, EVEX_BT8 | EVEX_ENDS | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vpermpd(ZMMRegister dst, const Operand &src, int8_t imm8, Mask mask = nomask) {
  zinstr(0x01, dst, src, imm8, mask, EVEX_BCST | EVEX_BT8 | EVEX_IMM | EVEX_L256 | EVEX_L512 | EVEX_M0F3A | EVEX_P66 | EVEX_W1);
}
void vpermpd(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x16, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT8 | EVEX_ENDS | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vpermps(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0x16, dst, src1, src2, 0, mask, EVEX_BT4 | EVEX_ENDS | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vpermps(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x16, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT4 | EVEX_ENDS | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vpermq(ZMMRegister dst, ZMMRegister src, int8_t imm8, Mask mask = nomask) {
  zinstr(0x00, dst, src, imm8, mask, EVEX_BT8 | EVEX_IMM | EVEX_L256 | EVEX_L512 | EVEX_M0F3A | EVEX_P66 | EVEX_W1);
}
void vpermq(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0x36, dst, src1, src2, 0, mask, EVEX_BT8 | EVEX_ENDS | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vpermq(ZMMRegister dst, const Operand &src, int8_t imm8, Mask mask = nomask) {
  zinstr(0x00, dst, src, imm8, mask, EVEX_BCST | EVEX_BT8 | EVEX_IMM | EVEX_L256 | EVEX_L512 | EVEX_M0F3A | EVEX_P66 | EVEX_W1);
}
void vpermq(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x36, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT8 | EVEX_ENDS | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vpermt2d(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0x7E, dst, src1, src2, 0, mask, EVEX_BT4 | EVEX_EDDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vpermt2d(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x7E, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT4 | EVEX_EDDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vpermt2pd(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0x7F, dst, src1, src2, 0, mask, EVEX_BT8 | EVEX_EDDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vpermt2pd(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x7F, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT8 | EVEX_EDDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vpermt2ps(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0x7F, dst, src1, src2, 0, mask, EVEX_BT4 | EVEX_EDDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vpermt2ps(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x7F, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT4 | EVEX_EDDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vpermt2q(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0x7E, dst, src1, src2, 0, mask, EVEX_BT8 | EVEX_EDDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vpermt2q(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x7E, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT8 | EVEX_EDDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vpexpandd(ZMMRegister dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x89, dst, src, 0, mask, EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vpexpandd(ZMMRegister dst, const Operand &src, Mask mask = nomask) {
  zinstr(0x89, dst, src, 0, mask, EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vpexpandq(ZMMRegister dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x89, dst, src, 0, mask, EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vpexpandq(ZMMRegister dst, const Operand &src, Mask mask = nomask) {
  zinstr(0x89, dst, src, 0, mask, EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vpmaxsb(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0x3C, dst, src1, src2, 0, mask, EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_WIG);
}
void vpmaxsb(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x3C, dst, src1, src2, 0, mask, EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_WIG);
}
void vpmaxsd(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0x3D, dst, src1, src2, 0, mask, EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vpmaxsd(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x3D, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vpmaxsq(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0x3D, dst, src1, src2, 0, mask, EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vpmaxsq(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x3D, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vpmaxsw(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0xEE, dst, src1, src2, 0, mask, EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_WIG);
}
void vpmaxsw(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0xEE, dst, src1, src2, 0, mask, EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_WIG);
}
void vpmaxud(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0x3F, dst, src1, src2, 0, mask, EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vpmaxud(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x3F, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vpmaxuq(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0x3F, dst, src1, src2, 0, mask, EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vpmaxuq(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x3F, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vpminsb(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0x38, dst, src1, src2, 0, mask, EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_WIG);
}
void vpminsb(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x38, dst, src1, src2, 0, mask, EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_WIG);
}
void vpminsd(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0x39, dst, src1, src2, 0, mask, EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vpminsd(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x39, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vpminsq(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0x39, dst, src1, src2, 0, mask, EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vpminsq(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x39, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vpminsw(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0xEA, dst, src1, src2, 0, mask, EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_WIG);
}
void vpminsw(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0xEA, dst, src1, src2, 0, mask, EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_WIG);
}
void vpminud(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0x3B, dst, src1, src2, 0, mask, EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vpminud(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x3B, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vpminuq(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0x3B, dst, src1, src2, 0, mask, EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vpminuq(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x3B, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vpmovb2m(OpmaskRegister dst, ZMMRegister src) {
  zinstr(0x29, dst, src, 0, nomask, EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_PF3 | EVEX_W0);
}
void vpmovd2m(OpmaskRegister dst, ZMMRegister src) {
  zinstr(0x39, dst, src, 0, nomask, EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_PF3 | EVEX_W0);
}
void vpmovdb(ZMMRegister dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x31, dst, src, 0, mask, EVEX_DT4 | EVEX_DT8 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_PF3 | EVEX_W0);
}
void vpmovdb(const Operand &dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x31, dst, src, 0, mask, EVEX_DT4 | EVEX_DT8 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_PF3 | EVEX_W0);
}
void vpmovdw(ZMMRegister dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x33, dst, src, 0, mask, EVEX_DT8 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_PF3 | EVEX_W0);
}
void vpmovdw(const Operand &dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x33, dst, src, 0, mask, EVEX_DT8 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_PF3 | EVEX_W0);
}
void vpmovm2b(ZMMRegister dst, OpmaskRegister src) {
  zinstr(0x28, dst, src, 0, nomask, EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_PF3 | EVEX_W0);
}
void vpmovm2d(ZMMRegister dst, OpmaskRegister src) {
  zinstr(0x38, dst, src, 0, nomask, EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_PF3 | EVEX_W0);
}
void vpmovm2q(ZMMRegister dst, OpmaskRegister src) {
  zinstr(0x38, dst, src, 0, nomask, EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_PF3 | EVEX_W1);
}
void vpmovm2w(ZMMRegister dst, OpmaskRegister src) {
  zinstr(0x28, dst, src, 0, nomask, EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_PF3 | EVEX_W1);
}
void vpmovq2m(OpmaskRegister dst, ZMMRegister src) {
  zinstr(0x39, dst, src, 0, nomask, EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_PF3 | EVEX_W1);
}
void vpmovqb(ZMMRegister dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x32, dst, src, 0, mask, EVEX_DT2 | EVEX_DT4 | EVEX_DT8 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_PF3 | EVEX_W0);
}
void vpmovqb(const Operand &dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x32, dst, src, 0, mask, EVEX_DT2 | EVEX_DT4 | EVEX_DT8 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_PF3 | EVEX_W0);
}
void vpmovqd(ZMMRegister dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x35, dst, src, 0, mask, EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_PF3 | EVEX_W0);
}
void vpmovqd(const Operand &dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x35, dst, src, 0, mask, EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_PF3 | EVEX_W0);
}
void vpmovqw(ZMMRegister dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x34, dst, src, 0, mask, EVEX_DT4 | EVEX_DT8 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_PF3 | EVEX_W0);
}
void vpmovqw(const Operand &dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x34, dst, src, 0, mask, EVEX_DT4 | EVEX_DT8 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_PF3 | EVEX_W0);
}
void vpmovsdb(ZMMRegister dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x21, dst, src, 0, mask, EVEX_DT4 | EVEX_DT8 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_PF3 | EVEX_W0);
}
void vpmovsdb(const Operand &dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x21, dst, src, 0, mask, EVEX_DT4 | EVEX_DT8 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_PF3 | EVEX_W0);
}
void vpmovsdw(ZMMRegister dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x23, dst, src, 0, mask, EVEX_DT8 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_PF3 | EVEX_W0);
}
void vpmovsdw(const Operand &dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x23, dst, src, 0, mask, EVEX_DT8 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_PF3 | EVEX_W0);
}
void vpmovsqb(ZMMRegister dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x22, dst, src, 0, mask, EVEX_DT2 | EVEX_DT4 | EVEX_DT8 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_PF3 | EVEX_W0);
}
void vpmovsqb(const Operand &dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x22, dst, src, 0, mask, EVEX_DT2 | EVEX_DT4 | EVEX_DT8 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_PF3 | EVEX_W0);
}
void vpmovsqd(ZMMRegister dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x25, dst, src, 0, mask, EVEX_DT8 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_PF3 | EVEX_W0);
}
void vpmovsqd(const Operand &dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x25, dst, src, 0, mask, EVEX_DT8 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_PF3 | EVEX_W0);
}
void vpmovsqw(ZMMRegister dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x24, dst, src, 0, mask, EVEX_DT4 | EVEX_DT8 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_PF3 | EVEX_W0);
}
void vpmovsqw(const Operand &dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x24, dst, src, 0, mask, EVEX_DT4 | EVEX_DT8 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_PF3 | EVEX_W0);
}
void vpmovsxbd(ZMMRegister dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x21, dst, src, 0, mask, EVEX_DT4 | EVEX_DT8 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_WIG);
}
void vpmovsxbd(ZMMRegister dst, const Operand &src, Mask mask = nomask) {
  zinstr(0x21, dst, src, 0, mask, EVEX_DT4 | EVEX_DT8 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_WIG);
}
void vpmovsxbq(ZMMRegister dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x22, dst, src, 0, mask, EVEX_DT2 | EVEX_DT4 | EVEX_DT8 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_WIG);
}
void vpmovsxbq(ZMMRegister dst, const Operand &src, Mask mask = nomask) {
  zinstr(0x22, dst, src, 0, mask, EVEX_DT2 | EVEX_DT4 | EVEX_DT8 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_WIG);
}
void vpmovsxdq(ZMMRegister dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x25, dst, src, 0, mask, EVEX_DT8 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vpmovsxdq(ZMMRegister dst, const Operand &src, Mask mask = nomask) {
  zinstr(0x25, dst, src, 0, mask, EVEX_DT8 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vpmovsxwd(ZMMRegister dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x23, dst, src, 0, mask, EVEX_DT8 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_WIG);
}
void vpmovsxwd(ZMMRegister dst, const Operand &src, Mask mask = nomask) {
  zinstr(0x23, dst, src, 0, mask, EVEX_DT8 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_WIG);
}
void vpmovsxwq(ZMMRegister dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x24, dst, src, 0, mask, EVEX_DT4 | EVEX_DT8 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_WIG);
}
void vpmovsxwq(ZMMRegister dst, const Operand &src, Mask mask = nomask) {
  zinstr(0x24, dst, src, 0, mask, EVEX_DT4 | EVEX_DT8 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_WIG);
}
void vpmovusdb(ZMMRegister dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x11, dst, src, 0, mask, EVEX_DT4 | EVEX_DT8 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_PF3 | EVEX_W0);
}
void vpmovusdb(const Operand &dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x11, dst, src, 0, mask, EVEX_DT4 | EVEX_DT8 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_PF3 | EVEX_W0);
}
void vpmovusdw(ZMMRegister dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x13, dst, src, 0, mask, EVEX_DT8 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_PF3 | EVEX_W0);
}
void vpmovusdw(const Operand &dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x13, dst, src, 0, mask, EVEX_DT8 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_PF3 | EVEX_W0);
}
void vpmovusqb(ZMMRegister dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x12, dst, src, 0, mask, EVEX_DT2 | EVEX_DT4 | EVEX_DT8 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_PF3 | EVEX_W0);
}
void vpmovusqb(const Operand &dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x12, dst, src, 0, mask, EVEX_DT2 | EVEX_DT4 | EVEX_DT8 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_PF3 | EVEX_W0);
}
void vpmovusqd(ZMMRegister dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x15, dst, src, 0, mask, EVEX_DT8 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_PF3 | EVEX_W0);
}
void vpmovusqd(const Operand &dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x15, dst, src, 0, mask, EVEX_DT8 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_PF3 | EVEX_W0);
}
void vpmovusqw(ZMMRegister dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x14, dst, src, 0, mask, EVEX_DT4 | EVEX_DT8 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_PF3 | EVEX_W0);
}
void vpmovusqw(const Operand &dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x14, dst, src, 0, mask, EVEX_DT4 | EVEX_DT8 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_PF3 | EVEX_W0);
}
void vpmovw2m(OpmaskRegister dst, ZMMRegister src) {
  zinstr(0x29, dst, src, 0, nomask, EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_PF3 | EVEX_W1);
}
void vpmovzxbd(ZMMRegister dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x31, dst, src, 0, mask, EVEX_DT4 | EVEX_DT8 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_WIG);
}
void vpmovzxbd(ZMMRegister dst, const Operand &src, Mask mask = nomask) {
  zinstr(0x31, dst, src, 0, mask, EVEX_DT4 | EVEX_DT8 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_WIG);
}
void vpmovzxbq(ZMMRegister dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x32, dst, src, 0, mask, EVEX_DT2 | EVEX_DT4 | EVEX_DT8 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_WIG);
}
void vpmovzxbq(ZMMRegister dst, const Operand &src, Mask mask = nomask) {
  zinstr(0x32, dst, src, 0, mask, EVEX_DT2 | EVEX_DT4 | EVEX_DT8 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_WIG);
}
void vpmovzxdq(ZMMRegister dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x35, dst, src, 0, mask, EVEX_DT8 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vpmovzxdq(ZMMRegister dst, const Operand &src, Mask mask = nomask) {
  zinstr(0x35, dst, src, 0, mask, EVEX_DT8 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vpmovzxwd(ZMMRegister dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x33, dst, src, 0, mask, EVEX_DT8 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_WIG);
}
void vpmovzxwd(ZMMRegister dst, const Operand &src, Mask mask = nomask) {
  zinstr(0x33, dst, src, 0, mask, EVEX_DT8 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_WIG);
}
void vpmovzxwq(ZMMRegister dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x34, dst, src, 0, mask, EVEX_DT4 | EVEX_DT8 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_WIG);
}
void vpmovzxwq(ZMMRegister dst, const Operand &src, Mask mask = nomask) {
  zinstr(0x34, dst, src, 0, mask, EVEX_DT4 | EVEX_DT8 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_WIG);
}
void vpmuldq(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0x28, dst, src1, src2, 0, mask, EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vpmuldq(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x28, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vpmulld(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0x40, dst, src1, src2, 0, mask, EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vpmulld(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x40, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vpmullq(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0x40, dst, src1, src2, 0, mask, EVEX_BT8 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vpmullq(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x40, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT8 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vpmullw(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0xD5, dst, src1, src2, 0, mask, EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_WIG);
}
void vpmullw(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0xD5, dst, src1, src2, 0, mask, EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_WIG);
}
void vpmuludq(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0xF4, dst, src1, src2, 0, mask, EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W1);
}
void vpmuludq(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0xF4, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W1);
}
void vpord(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0xEB, dst, src1, src2, 0, mask, EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W0);
}
void vpord(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0xEB, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W0);
}
void vporq(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0xEB, dst, src1, src2, 0, mask, EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W1);
}
void vporq(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0xEB, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W1);
}
void vprold(ZMMRegister dst, ZMMRegister src, int8_t imm8, Mask mask = nomask) {
  zinstr(0x72, zmm1, dst, src, imm8, mask, EVEX_BT4 | EVEX_ENDD | EVEX_IMM | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W0);
}
void vprold(ZMMRegister dst, const Operand &src, int8_t imm8, Mask mask = nomask) {
  zinstr(0x72, zmm1, dst, src, imm8, mask, EVEX_BCST | EVEX_BT4 | EVEX_ENDD | EVEX_IMM | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W0);
}
void vprolq(ZMMRegister dst, ZMMRegister src, int8_t imm8, Mask mask = nomask) {
  zinstr(0x72, zmm1, dst, src, imm8, mask, EVEX_BT8 | EVEX_ENDD | EVEX_IMM | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W1);
}
void vprolq(ZMMRegister dst, const Operand &src, int8_t imm8, Mask mask = nomask) {
  zinstr(0x72, zmm1, dst, src, imm8, mask, EVEX_BCST | EVEX_BT8 | EVEX_ENDD | EVEX_IMM | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W1);
}
void vprolvd(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0x15, dst, src1, src2, 0, mask, EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vprolvd(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x15, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vprolvq(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0x15, dst, src1, src2, 0, mask, EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vprolvq(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x15, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vprord(ZMMRegister dst, ZMMRegister src, int8_t imm8, Mask mask = nomask) {
  zinstr(0x72, zmm0, dst, src, imm8, mask, EVEX_BT4 | EVEX_ENDD | EVEX_IMM | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W0);
}
void vprord(ZMMRegister dst, const Operand &src, int8_t imm8, Mask mask = nomask) {
  zinstr(0x72, zmm0, dst, src, imm8, mask, EVEX_BCST | EVEX_BT4 | EVEX_ENDD | EVEX_IMM | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W0);
}
void vprorq(ZMMRegister dst, ZMMRegister src, int8_t imm8, Mask mask = nomask) {
  zinstr(0x72, zmm0, dst, src, imm8, mask, EVEX_BT8 | EVEX_ENDD | EVEX_IMM | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W1);
}
void vprorq(ZMMRegister dst, const Operand &src, int8_t imm8, Mask mask = nomask) {
  zinstr(0x72, zmm0, dst, src, imm8, mask, EVEX_BCST | EVEX_BT8 | EVEX_ENDD | EVEX_IMM | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W1);
}
void vprorvd(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0x14, dst, src1, src2, 0, mask, EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vprorvd(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x14, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vprorvq(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0x14, dst, src1, src2, 0, mask, EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vprorvq(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x14, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vpshufd(ZMMRegister dst, ZMMRegister src, int8_t imm8, Mask mask = nomask) {
  zinstr(0x70, dst, src, imm8, mask, EVEX_BT4 | EVEX_IMM | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W0);
}
void vpshufd(ZMMRegister dst, const Operand &src, int8_t imm8, Mask mask = nomask) {
  zinstr(0x70, dst, src, imm8, mask, EVEX_BCST | EVEX_BT4 | EVEX_IMM | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W0);
}
void vpslld(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0xF2, dst, src1, src2, 0, mask, EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W0);
}
void vpslld(ZMMRegister dst, ZMMRegister src, int8_t imm8, Mask mask = nomask) {
  zinstr(0x72, zmm6, dst, src, imm8, mask, EVEX_BT4 | EVEX_ENDD | EVEX_IMM | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W0);
}
void vpslld(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0xF2, dst, src1, src2, 0, mask, EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W0);
}
void vpslld(ZMMRegister dst, const Operand &src, int8_t imm8, Mask mask = nomask) {
  zinstr(0x72, zmm6, dst, src, imm8, mask, EVEX_BCST | EVEX_BT4 | EVEX_ENDD | EVEX_IMM | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W0);
}
void vpsllq(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0xF3, dst, src1, src2, 0, mask, EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W1);
}
void vpsllq(ZMMRegister dst, ZMMRegister src, int8_t imm8, Mask mask = nomask) {
  zinstr(0x73, zmm6, dst, src, imm8, mask, EVEX_BT8 | EVEX_ENDD | EVEX_IMM | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W1);
}
void vpsllq(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0xF3, dst, src1, src2, 0, mask, EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W1);
}
void vpsllq(ZMMRegister dst, const Operand &src, int8_t imm8, Mask mask = nomask) {
  zinstr(0x73, zmm6, dst, src, imm8, mask, EVEX_BCST | EVEX_BT8 | EVEX_ENDD | EVEX_IMM | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W1);
}
void vpsllvd(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0x47, dst, src1, src2, 0, mask, EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vpsllvd(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x47, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vpsllvq(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0x47, dst, src1, src2, 0, mask, EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vpsllvq(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x47, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vpsllw(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0xF1, dst, src1, src2, 0, mask, EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_WIG);
}
void vpsllw(ZMMRegister dst, ZMMRegister src, int8_t imm8, Mask mask = nomask) {
  zinstr(0x71, zmm6, dst, src, imm8, mask, EVEX_ENDD | EVEX_IMM | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_WIG);
}
void vpsllw(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0xF1, dst, src1, src2, 0, mask, EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_WIG);
}
void vpsllw(ZMMRegister dst, const Operand &src, int8_t imm8, Mask mask = nomask) {
  zinstr(0x71, zmm6, dst, src, imm8, mask, EVEX_ENDD | EVEX_IMM | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_WIG);
}
void vpsravd(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0x46, dst, src1, src2, 0, mask, EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vpsravd(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x46, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vpsravq(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0x46, dst, src1, src2, 0, mask, EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vpsravq(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x46, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vpsrld(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0xD2, dst, src1, src2, 0, mask, EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W0);
}
void vpsrld(ZMMRegister dst, ZMMRegister src, int8_t imm8, Mask mask = nomask) {
  zinstr(0x72, zmm2, dst, src, imm8, mask, EVEX_BT4 | EVEX_ENDD | EVEX_IMM | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W0);
}
void vpsrld(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0xD2, dst, src1, src2, 0, mask, EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W0);
}
void vpsrld(ZMMRegister dst, const Operand &src, int8_t imm8, Mask mask = nomask) {
  zinstr(0x72, zmm2, dst, src, imm8, mask, EVEX_BCST | EVEX_BT4 | EVEX_ENDD | EVEX_IMM | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W0);
}
void vpsrlq(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0xD3, dst, src1, src2, 0, mask, EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W1);
}
void vpsrlq(ZMMRegister dst, ZMMRegister src, int8_t imm8, Mask mask = nomask) {
  zinstr(0x73, zmm2, dst, src, imm8, mask, EVEX_BT8 | EVEX_ENDD | EVEX_IMM | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W1);
}
void vpsrlq(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0xD3, dst, src1, src2, 0, mask, EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W1);
}
void vpsrlq(ZMMRegister dst, const Operand &src, int8_t imm8, Mask mask = nomask) {
  zinstr(0x73, zmm2, dst, src, imm8, mask, EVEX_BCST | EVEX_BT8 | EVEX_ENDD | EVEX_IMM | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W1);
}
void vpsrlvd(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0x45, dst, src1, src2, 0, mask, EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vpsrlvd(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x45, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vpsrlvq(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0x45, dst, src1, src2, 0, mask, EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vpsrlvq(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x45, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vpsrlw(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0xD1, dst, src1, src2, 0, mask, EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_WIG);
}
void vpsrlw(ZMMRegister dst, ZMMRegister src, int8_t imm8, Mask mask = nomask) {
  zinstr(0x71, zmm2, dst, src, imm8, mask, EVEX_ENDD | EVEX_IMM | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_WIG);
}
void vpsrlw(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0xD1, dst, src1, src2, 0, mask, EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_WIG);
}
void vpsrlw(ZMMRegister dst, const Operand &src, int8_t imm8, Mask mask = nomask) {
  zinstr(0x71, zmm2, dst, src, imm8, mask, EVEX_ENDD | EVEX_IMM | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_WIG);
}
void vpsubb(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0xF8, dst, src1, src2, 0, mask, EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_WIG);
}
void vpsubb(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0xF8, dst, src1, src2, 0, mask, EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_WIG);
}
void vpsubd(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0xFA, dst, src1, src2, 0, mask, EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W0);
}
void vpsubd(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0xFA, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W0);
}
void vpsubq(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0xFB, dst, src1, src2, 0, mask, EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W1);
}
void vpsubq(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0xFB, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W1);
}
void vpsubw(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0xF9, dst, src1, src2, 0, mask, EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_WIG);
}
void vpsubw(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0xF9, dst, src1, src2, 0, mask, EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_WIG);
}
void vpternlogd(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, int8_t imm8, Mask mask = nomask) {
  zinstr(0x25, dst, src1, src2, imm8, mask, EVEX_BT4 | EVEX_EDDS | EVEX_IMM | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F3A | EVEX_P66 | EVEX_W0);
}
void vpternlogd(ZMMRegister dst, ZMMRegister src1, const Operand &src2, int8_t imm8, Mask mask = nomask) {
  zinstr(0x25, dst, src1, src2, imm8, mask, EVEX_BCST | EVEX_BT4 | EVEX_EDDS | EVEX_IMM | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F3A | EVEX_P66 | EVEX_W0);
}
void vpternlogq(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, int8_t imm8, Mask mask = nomask) {
  zinstr(0x25, dst, src1, src2, imm8, mask, EVEX_BT8 | EVEX_EDDS | EVEX_IMM | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F3A | EVEX_P66 | EVEX_W1);
}
void vpternlogq(ZMMRegister dst, ZMMRegister src1, const Operand &src2, int8_t imm8, Mask mask = nomask) {
  zinstr(0x25, dst, src1, src2, imm8, mask, EVEX_BCST | EVEX_BT8 | EVEX_EDDS | EVEX_IMM | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F3A | EVEX_P66 | EVEX_W1);
}
void vptestmd(OpmaskRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0x27, dst, src1, src2, 0, mask, EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vptestmd(OpmaskRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x27, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vptestmq(OpmaskRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0x27, dst, src1, src2, 0, mask, EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vptestmq(OpmaskRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x27, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vptestnmb(OpmaskRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0x26, dst, src1, src2, 0, mask, EVEX_ENDS | EVEX_L512 | EVEX_M0F38 | EVEX_PF3 | EVEX_W0);
}
void vptestnmb(OpmaskRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x26, dst, src1, src2, 0, mask, EVEX_ENDS | EVEX_L512 | EVEX_M0F38 | EVEX_PF3 | EVEX_W0);
}
void vptestnmd(OpmaskRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0x27, dst, src1, src2, 0, mask, EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_PF3 | EVEX_W0);
}
void vptestnmd(OpmaskRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x27, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_PF3 | EVEX_W0);
}
void vptestnmq(OpmaskRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0x27, dst, src1, src2, 0, mask, EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_PF3 | EVEX_W1);
}
void vptestnmq(OpmaskRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x27, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_PF3 | EVEX_W1);
}
void vptestnmw(OpmaskRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0x26, dst, src1, src2, 0, mask, EVEX_ENDS | EVEX_L512 | EVEX_M0F38 | EVEX_PF3 | EVEX_W1);
}
void vptestnmw(OpmaskRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x26, dst, src1, src2, 0, mask, EVEX_ENDS | EVEX_L512 | EVEX_M0F38 | EVEX_PF3 | EVEX_W1);
}
void vpunpckhdq(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0x6A, dst, src1, src2, 0, mask, EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_M0F | EVEX_P66 | EVEX_W0);
}
void vpunpckhdq(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x6A, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_M0F | EVEX_P66 | EVEX_W0);
}
void vpunpckhqdq(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0x6D, dst, src1, src2, 0, mask, EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_M0F | EVEX_P66 | EVEX_W1);
}
void vpunpckhqdq(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x6D, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_M0F | EVEX_P66 | EVEX_W1);
}
void vpunpckldq(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0x62, dst, src1, src2, 0, mask, EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_M0F | EVEX_P66 | EVEX_W0);
}
void vpunpckldq(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x62, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_M0F | EVEX_P66 | EVEX_W0);
}
void vpunpcklqdq(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0x6C, dst, src1, src2, 0, mask, EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_M0F | EVEX_P66 | EVEX_W1);
}
void vpunpcklqdq(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x6C, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_M0F | EVEX_P66 | EVEX_W1);
}
void vpxord(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0xEF, dst, src1, src2, 0, mask, EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W0);
}
void vpxord(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0xEF, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W0);
}
void vpxorq(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0xEF, dst, src1, src2, 0, mask, EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W1);
}
void vpxorq(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0xEF, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W1);
}
void vrcp14pd(ZMMRegister dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x4C, dst, src, 0, mask, EVEX_BT8 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vrcp14pd(ZMMRegister dst, const Operand &src, Mask mask = nomask) {
  zinstr(0x4C, dst, src, 0, mask, EVEX_BCST | EVEX_BT8 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vrcp14ps(ZMMRegister dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x4C, dst, src, 0, mask, EVEX_BT4 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vrcp14ps(ZMMRegister dst, const Operand &src, Mask mask = nomask) {
  zinstr(0x4C, dst, src, 0, mask, EVEX_BCST | EVEX_BT4 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vrcp14sd(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0x4D, dst, src1, src2, 0, mask, EVEX_DT8 | EVEX_ENDS | EVEX_LIG | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vrcp14sd(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x4D, dst, src1, src2, 0, mask, EVEX_DT8 | EVEX_ENDS | EVEX_LIG | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vrcp14ss(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0x4D, dst, src1, src2, 0, mask, EVEX_DT4 | EVEX_ENDS | EVEX_LIG | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vrcp14ss(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x4D, dst, src1, src2, 0, mask, EVEX_DT4 | EVEX_ENDS | EVEX_LIG | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vrndscalepd(ZMMRegister dst, ZMMRegister src, int8_t imm8, Mask mask = nomask) {
  zinstr(0x09, dst, src, imm8, mask, EVEX_BT8 | EVEX_IMM | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F3A | EVEX_P66 | EVEX_SAE | EVEX_W1);
}
void vrndscalepd(ZMMRegister dst, const Operand &src, int8_t imm8, Mask mask = nomask) {
  zinstr(0x09, dst, src, imm8, mask, EVEX_BCST | EVEX_BT8 | EVEX_IMM | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F3A | EVEX_P66 | EVEX_W1);
}
void vrndscaleps(ZMMRegister dst, ZMMRegister src, int8_t imm8, Mask mask = nomask) {
  zinstr(0x08, dst, src, imm8, mask, EVEX_BT4 | EVEX_IMM | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F3A | EVEX_P66 | EVEX_SAE | EVEX_W0);
}
void vrndscaleps(ZMMRegister dst, const Operand &src, int8_t imm8, Mask mask = nomask) {
  zinstr(0x08, dst, src, imm8, mask, EVEX_BCST | EVEX_BT4 | EVEX_IMM | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F3A | EVEX_P66 | EVEX_W0);
}
void vrndscalesd(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, int8_t imm8, Mask mask = nomask) {
  zinstr(0x0B, dst, src1, src2, imm8, mask, EVEX_DT8 | EVEX_ENDS | EVEX_IMM | EVEX_LIG | EVEX_M0F3A | EVEX_P66 | EVEX_SAE | EVEX_W1);
}
void vrndscalesd(ZMMRegister dst, ZMMRegister src1, const Operand &src2, int8_t imm8, Mask mask = nomask) {
  zinstr(0x0B, dst, src1, src2, imm8, mask, EVEX_DT8 | EVEX_ENDS | EVEX_IMM | EVEX_LIG | EVEX_M0F3A | EVEX_P66 | EVEX_SAE | EVEX_W1);
}
void vrndscaless(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, int8_t imm8, Mask mask = nomask) {
  zinstr(0x0A, dst, src1, src2, imm8, mask, EVEX_DT4 | EVEX_ENDS | EVEX_IMM | EVEX_LIG | EVEX_M0F3A | EVEX_P66 | EVEX_SAE | EVEX_W0);
}
void vrndscaless(ZMMRegister dst, ZMMRegister src1, const Operand &src2, int8_t imm8, Mask mask = nomask) {
  zinstr(0x0A, dst, src1, src2, imm8, mask, EVEX_DT4 | EVEX_ENDS | EVEX_IMM | EVEX_LIG | EVEX_M0F3A | EVEX_P66 | EVEX_SAE | EVEX_W0);
}
void vrsqrt14pd(ZMMRegister dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x4E, dst, src, 0, mask, EVEX_BT8 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vrsqrt14pd(ZMMRegister dst, const Operand &src, Mask mask = nomask) {
  zinstr(0x4E, dst, src, 0, mask, EVEX_BCST | EVEX_BT8 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vrsqrt14ps(ZMMRegister dst, ZMMRegister src, Mask mask = nomask) {
  zinstr(0x4E, dst, src, 0, mask, EVEX_BT4 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vrsqrt14ps(ZMMRegister dst, const Operand &src, Mask mask = nomask) {
  zinstr(0x4E, dst, src, 0, mask, EVEX_BCST | EVEX_BT4 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vrsqrt14sd(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0x4F, dst, src1, src2, 0, mask, EVEX_DT8 | EVEX_ENDS | EVEX_LIG | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vrsqrt14sd(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x4F, dst, src1, src2, 0, mask, EVEX_DT8 | EVEX_ENDS | EVEX_LIG | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vrsqrt14ss(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0x4F, dst, src1, src2, 0, mask, EVEX_DT4 | EVEX_ENDS | EVEX_LIG | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vrsqrt14ss(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x4F, dst, src1, src2, 0, mask, EVEX_DT4 | EVEX_ENDS | EVEX_LIG | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vscalefpd(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0x2C, dst, src1, src2, 0, mask, EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1 | evex_round(er));
}
void vscalefpd(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x2C, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vscalefps(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0x2C, dst, src1, src2, 0, mask, EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0 | evex_round(er));
}
void vscalefps(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x2C, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vscalefsd(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0x2D, dst, src1, src2, 0, mask, EVEX_DT8 | EVEX_ENDS | EVEX_LIG | EVEX_M0F38 | EVEX_P66 | EVEX_W1 | evex_round(er));
}
void vscalefsd(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x2D, dst, src1, src2, 0, mask, EVEX_DT8 | EVEX_ENDS | EVEX_LIG | EVEX_M0F38 | EVEX_P66 | EVEX_W1);
}
void vscalefss(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0x2D, dst, src1, src2, 0, mask, EVEX_DT4 | EVEX_ENDS | EVEX_LIG | EVEX_M0F38 | EVEX_P66 | EVEX_W0 | evex_round(er));
}
void vscalefss(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x2D, dst, src1, src2, 0, mask, EVEX_DT4 | EVEX_ENDS | EVEX_LIG | EVEX_M0F38 | EVEX_P66 | EVEX_W0);
}
void vshuff32x4(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, int8_t imm8, Mask mask = nomask) {
  zinstr(0x23, dst, src1, src2, imm8, mask, EVEX_BT4 | EVEX_ENDS | EVEX_IMM | EVEX_L256 | EVEX_L512 | EVEX_M0F3A | EVEX_P66 | EVEX_W0);
}
void vshuff32x4(ZMMRegister dst, ZMMRegister src1, const Operand &src2, int8_t imm8, Mask mask = nomask) {
  zinstr(0x23, dst, src1, src2, imm8, mask, EVEX_BCST | EVEX_BT4 | EVEX_ENDS | EVEX_IMM | EVEX_L256 | EVEX_L512 | EVEX_M0F3A | EVEX_P66 | EVEX_W0);
}
void vshuff64x2(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, int8_t imm8, Mask mask = nomask) {
  zinstr(0x23, dst, src1, src2, imm8, mask, EVEX_BT8 | EVEX_ENDS | EVEX_IMM | EVEX_L256 | EVEX_L512 | EVEX_M0F3A | EVEX_P66 | EVEX_W1);
}
void vshuff64x2(ZMMRegister dst, ZMMRegister src1, const Operand &src2, int8_t imm8, Mask mask = nomask) {
  zinstr(0x23, dst, src1, src2, imm8, mask, EVEX_BCST | EVEX_BT8 | EVEX_ENDS | EVEX_IMM | EVEX_L256 | EVEX_L512 | EVEX_M0F3A | EVEX_P66 | EVEX_W1);
}
void vshufi32x4(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, int8_t imm8, Mask mask = nomask) {
  zinstr(0x43, dst, src1, src2, imm8, mask, EVEX_BT4 | EVEX_ENDS | EVEX_IMM | EVEX_L256 | EVEX_L512 | EVEX_M0F3A | EVEX_P66 | EVEX_W0);
}
void vshufi32x4(ZMMRegister dst, ZMMRegister src1, const Operand &src2, int8_t imm8, Mask mask = nomask) {
  zinstr(0x43, dst, src1, src2, imm8, mask, EVEX_BCST | EVEX_BT4 | EVEX_ENDS | EVEX_IMM | EVEX_L256 | EVEX_L512 | EVEX_M0F3A | EVEX_P66 | EVEX_W0);
}
void vshufi64x2(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, int8_t imm8, Mask mask = nomask) {
  zinstr(0x43, dst, src1, src2, imm8, mask, EVEX_BT8 | EVEX_ENDS | EVEX_IMM | EVEX_L256 | EVEX_L512 | EVEX_M0F3A | EVEX_P66 | EVEX_W1);
}
void vshufi64x2(ZMMRegister dst, ZMMRegister src1, const Operand &src2, int8_t imm8, Mask mask = nomask) {
  zinstr(0x43, dst, src1, src2, imm8, mask, EVEX_BCST | EVEX_BT8 | EVEX_ENDS | EVEX_IMM | EVEX_L256 | EVEX_L512 | EVEX_M0F3A | EVEX_P66 | EVEX_W1);
}
void vshufpd(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, int8_t imm8, Mask mask = nomask) {
  zinstr(0xC6, dst, src1, src2, imm8, mask, EVEX_BT8 | EVEX_ENDS | EVEX_IMM | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W1);
}
void vshufpd(ZMMRegister dst, ZMMRegister src1, const Operand &src2, int8_t imm8, Mask mask = nomask) {
  zinstr(0xC6, dst, src1, src2, imm8, mask, EVEX_BCST | EVEX_BT8 | EVEX_ENDS | EVEX_IMM | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W1);
}
void vshufps(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, int8_t imm8, Mask mask = nomask) {
  zinstr(0xC6, dst, src1, src2, imm8, mask, EVEX_BT4 | EVEX_ENDS | EVEX_IMM | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_W0);
}
void vshufps(ZMMRegister dst, ZMMRegister src1, const Operand &src2, int8_t imm8, Mask mask = nomask) {
  zinstr(0xC6, dst, src1, src2, imm8, mask, EVEX_BCST | EVEX_BT4 | EVEX_ENDS | EVEX_IMM | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_W0);
}
void vsqrtpd(ZMMRegister dst, ZMMRegister src, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0x51, dst, src, 0, mask, EVEX_BT8 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W1 | evex_round(er));
}
void vsqrtpd(ZMMRegister dst, const Operand &src, Mask mask = nomask) {
  zinstr(0x51, dst, src, 0, mask, EVEX_BCST | EVEX_BT8 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W1);
}
void vsqrtps(ZMMRegister dst, ZMMRegister src, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0x51, dst, src, 0, mask, EVEX_BT4 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_W0 | evex_round(er));
}
void vsqrtps(ZMMRegister dst, const Operand &src, Mask mask = nomask) {
  zinstr(0x51, dst, src, 0, mask, EVEX_BCST | EVEX_BT4 | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_W0);
}
void vsqrtsd(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0x51, dst, src1, src2, 0, mask, EVEX_DT8 | EVEX_ENDS | EVEX_LIG | EVEX_M0F | EVEX_PF2 | EVEX_W1 | evex_round(er));
}
void vsqrtsd(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x51, dst, src1, src2, 0, mask, EVEX_DT8 | EVEX_ENDS | EVEX_LIG | EVEX_M0F | EVEX_PF2 | EVEX_W1);
}
void vsqrtss(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0x51, dst, src1, src2, 0, mask, EVEX_DT4 | EVEX_ENDS | EVEX_LIG | EVEX_M0F | EVEX_PF3 | EVEX_W0 | evex_round(er));
}
void vsqrtss(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x51, dst, src1, src2, 0, mask, EVEX_DT4 | EVEX_ENDS | EVEX_LIG | EVEX_M0F | EVEX_PF3 | EVEX_W0);
}
void vsubpd(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0x5C, dst, src1, src2, 0, mask, EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W1 | evex_round(er));
}
void vsubpd(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x5C, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W1);
}
void vsubps(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0x5C, dst, src1, src2, 0, mask, EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_W0 | evex_round(er));
}
void vsubps(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x5C, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_W0);
}
void vsubsd(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0x5C, dst, src1, src2, 0, mask, EVEX_DT8 | EVEX_ENDS | EVEX_LIG | EVEX_M0F | EVEX_PF2 | EVEX_W1 | evex_round(er));
}
void vsubsd(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x5C, dst, src1, src2, 0, mask, EVEX_DT8 | EVEX_ENDS | EVEX_LIG | EVEX_M0F | EVEX_PF2 | EVEX_W1);
}
void vsubss(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask, RoundingMode er = noround) {
  zinstr(0x5C, dst, src1, src2, 0, mask, EVEX_DT4 | EVEX_ENDS | EVEX_LIG | EVEX_M0F | EVEX_PF3 | EVEX_W0 | evex_round(er));
}
void vsubss(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x5C, dst, src1, src2, 0, mask, EVEX_DT4 | EVEX_ENDS | EVEX_LIG | EVEX_M0F | EVEX_PF3 | EVEX_W0);
}
void vucomisd(ZMMRegister dst, ZMMRegister src) {
  zinstr(0x2E, dst, src, 0, nomask, EVEX_DT8 | EVEX_LIG | EVEX_M0F | EVEX_P66 | EVEX_SAE | EVEX_W1);
}
void vucomisd(ZMMRegister dst, const Operand &src) {
  zinstr(0x2E, dst, src, 0, nomask, EVEX_DT8 | EVEX_LIG | EVEX_M0F | EVEX_P66 | EVEX_SAE | EVEX_W1);
}
void vucomiss(ZMMRegister dst, ZMMRegister src) {
  zinstr(0x2E, dst, src, 0, nomask, EVEX_DT4 | EVEX_LIG | EVEX_M0F | EVEX_SAE | EVEX_W0);
}
void vucomiss(ZMMRegister dst, const Operand &src) {
  zinstr(0x2E, dst, src, 0, nomask, EVEX_DT4 | EVEX_LIG | EVEX_M0F | EVEX_SAE | EVEX_W0);
}
void vunpckhpd(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0x15, dst, src1, src2, 0, mask, EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W1);
}
void vunpckhpd(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x15, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W1);
}
void vunpckhps(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0x15, dst, src1, src2, 0, mask, EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_W0);
}
void vunpckhps(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x15, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_W0);
}
void vunpcklpd(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0x14, dst, src1, src2, 0, mask, EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W1);
}
void vunpcklpd(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x14, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W1);
}
void vunpcklps(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0x14, dst, src1, src2, 0, mask, EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_W0);
}
void vunpcklps(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x14, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_W0);
}
void vxorpd(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0x57, dst, src1, src2, 0, mask, EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W1);
}
void vxorpd(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x57, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT8 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_P66 | EVEX_W1);
}
void vxorps(ZMMRegister dst, ZMMRegister src1, ZMMRegister src2, Mask mask = nomask) {
  zinstr(0x57, dst, src1, src2, 0, mask, EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_W0);
}
void vxorps(ZMMRegister dst, ZMMRegister src1, const Operand &src2, Mask mask = nomask) {
  zinstr(0x57, dst, src1, src2, 0, mask, EVEX_BCST | EVEX_BT4 | EVEX_ENDS | EVEX_L128 | EVEX_L256 | EVEX_L512 | EVEX_M0F | EVEX_W0);
}
