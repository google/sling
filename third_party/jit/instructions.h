// Copyright (c) 1994-2006 Sun Microsystems Inc.
// All Rights Reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// - Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// - Redistribution in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// - Neither the name of Sun Microsystems or the names of contributors may
// be used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
// IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// The original source code covered by the above license above has been
// modified significantly by Google Inc.
// Copyright 2012 the V8 project authors. All rights reserved.
// Copyright 2017 Google Inc. All rights reserved.

#ifndef JIT_INSTRUCTIONS_H_
#define JIT_INSTRUCTIONS_H_

namespace sling {
namespace jit {

// General instructions.
#define ASSEMBLER_INSTRUCTION_LIST(V) \
  V(add)                              \
  V(and)                              \
  V(cmp)                              \
  V(cmpxchg)                          \
  V(dec)                              \
  V(idiv)                             \
  V(div)                              \
  V(imul)                             \
  V(inc)                              \
  V(lea)                              \
  V(mov)                              \
  V(movzxb)                           \
  V(movzxw)                           \
  V(neg)                              \
  V(not)                              \
  V(or)                               \
  V(repmovs)                          \
  V(sbb)                              \
  V(sub)                              \
  V(test)                             \
  V(xchg)                             \
  V(xor)

// Shift instructions on operands/registers with pointer, 32-bit and 64-bit.
#define SHIFT_INSTRUCTION_LIST(V)       \
  V(rol, 0x0)                           \
  V(ror, 0x1)                           \
  V(rcl, 0x2)                           \
  V(rcr, 0x3)                           \
  V(shl, 0x4)                           \
  V(shr, 0x5)                           \
  V(sal, 0x4)                           \
  V(sar, 0x7)                           \

// SSE instructions.
#define SSE2_INSTRUCTION_LIST(V) \
  V(packsswb, 66, 0F, 63)        \
  V(packssdw, 66, 0F, 6B)        \
  V(packuswb, 66, 0F, 67)        \
  V(paddb, 66, 0F, FC)           \
  V(paddw, 66, 0F, FD)           \
  V(paddd, 66, 0F, FE)           \
  V(paddq, 66, 0F, D4)           \
  V(paddsb, 66, 0F, EC)          \
  V(paddsw, 66, 0F, ED)          \
  V(paddusb, 66, 0F, DC)         \
  V(paddusw, 66, 0F, DD)         \
  V(pcmpeqb, 66, 0F, 74)         \
  V(pcmpeqw, 66, 0F, 75)         \
  V(pcmpeqd, 66, 0F, 76)         \
  V(pcmpgtb, 66, 0F, 64)         \
  V(pcmpgtw, 66, 0F, 65)         \
  V(pcmpgtd, 66, 0F, 66)         \
  V(pmaxsw, 66, 0F, EE)          \
  V(pmaxub, 66, 0F, DE)          \
  V(pminsw, 66, 0F, EA)          \
  V(pminub, 66, 0F, DA)          \
  V(pmullw, 66, 0F, D5)          \
  V(pmuludq, 66, 0F, F4)         \
  V(psllw, 66, 0F, F1)           \
  V(pslld, 66, 0F, F2)           \
  V(psraw, 66, 0F, E1)           \
  V(psrad, 66, 0F, E2)           \
  V(psrlw, 66, 0F, D1)           \
  V(psrld, 66, 0F, D2)           \
  V(psubb, 66, 0F, F8)           \
  V(psubw, 66, 0F, F9)           \
  V(psubd, 66, 0F, FA)           \
  V(psubq, 66, 0F, FB)           \
  V(psubsb, 66, 0F, E8)          \
  V(psubsw, 66, 0F, E9)          \
  V(psubusb, 66, 0F, D8)         \
  V(psubusw, 66, 0F, D9)         \
  V(pxor, 66, 0F, EF)            \
  V(pand, 66, 0F, DB)            \
  V(por, 66, 0F, EB)             \
  V(cvtps2dq, 66, 0F, 5B)

#define SSSE3_INSTRUCTION_LIST(V) \
  V(pabsb, 66, 0F, 38, 1C)        \
  V(pabsw, 66, 0F, 38, 1D)        \
  V(pabsd, 66, 0F, 38, 1E)        \
  V(phaddd, 66, 0F, 38, 02)       \
  V(phaddsw, 66, 0F, 38, 03)      \
  V(phaddw, 66, 0F, 38, 01)       \
  V(pmaddubsw, 66, 0F, 38, 04)    \
  V(pshufb, 66, 0F, 38, 00)       \
  V(psignb, 66, 0F, 38, 08)       \
  V(psignw, 66, 0F, 38, 09)       \
  V(psignd, 66, 0F, 38, 0A)

#define SSE4_INSTRUCTION_LIST(V) \
  V(packusdw, 66, 0F, 38, 2B)    \
  V(pminsb, 66, 0F, 38, 38)      \
  V(pminsd, 66, 0F, 38, 39)      \
  V(pminuw, 66, 0F, 38, 3A)      \
  V(pminud, 66, 0F, 38, 3B)      \
  V(pmaxsb, 66, 0F, 38, 3C)      \
  V(pmaxsd, 66, 0F, 38, 3D)      \
  V(pmaxuw, 66, 0F, 38, 3E)      \
  V(pmaxud, 66, 0F, 38, 3F)      \
  V(pmulld, 66, 0F, 38, 40)      \
  V(pcmpeqq, 66, 0F, 38, 29)

}  // namespace jit
}  // namespace sling

#endif  // JIT_INSTRUCTIONS_H_
