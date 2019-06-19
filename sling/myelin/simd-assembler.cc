// Copyright 2017 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <limits>

#include "sling/myelin/simd-assembler.h"

namespace sling {
namespace myelin {

using namespace jit;

template<typename T> StaticData *MinVal(MacroAssembler *masm, int repeat) {
  return masm->GetConstant<T>(std::numeric_limits<T>::min(), repeat);
}

template<typename T> StaticData *MaxVal(MacroAssembler *masm, int repeat) {
  return masm->GetConstant<T>(std::numeric_limits<T>::max(), repeat);
}

StaticData *SIMDGenerator::NeutralElement(Reduction op, Type type, int repeat) {
  switch (op) {
    case REDUCE_ADD:
      return nullptr;
    case REDUCE_MUL:
      switch (type) {
        case DT_FLOAT: return masm_->GetConstant<float>(1.0, repeat);
        case DT_DOUBLE: return masm_->GetConstant<double>(1.0, repeat);
        case DT_INT8: return masm_->GetConstant<int8>(1, repeat);
        case DT_INT16: return masm_->GetConstant<int16>(1, repeat);
        case DT_INT32: return masm_->GetConstant<int32>(1, repeat);
        case DT_INT64: return masm_->GetConstant<int64>(1, repeat);
        default: return nullptr;
      }
    case REDUCE_MIN:
      switch (type) {
        case DT_FLOAT: return MaxVal<float>(masm_, repeat);
        case DT_DOUBLE: return MaxVal<double>(masm_, repeat);
        case DT_INT8: return MaxVal<int8>(masm_, repeat);
        case DT_INT16: return MaxVal<int16>(masm_, repeat);
        case DT_INT32: return MaxVal<int32>(masm_, repeat);
        case DT_INT64: return MaxVal<int64>(masm_, repeat);
        default: return nullptr;
      }
    case REDUCE_MAX:
      switch (type) {
        case DT_FLOAT: return MinVal<float>(masm_, repeat);
        case DT_DOUBLE: return MinVal<double>(masm_, repeat);
        case DT_INT8: return MinVal<int8>(masm_, repeat);
        case DT_INT16: return MinVal<int16>(masm_, repeat);
        case DT_INT32: return MinVal<int32>(masm_, repeat);
        case DT_INT64: return MinVal<int64>(masm_, repeat);
        default: return nullptr;
      }
    case REDUCE_AND:
      switch (type) {
        case DT_FLOAT: return masm_->GetConstant<int32>(-1, repeat);
        case DT_DOUBLE: return masm_->GetConstant<int64>(-1, repeat);
        case DT_INT8: return masm_->GetConstant<int8>(-1, repeat);
        case DT_INT16: return masm_->GetConstant<int16>(-1, repeat);
        case DT_INT32: return masm_->GetConstant<int32>(-1, repeat);
        case DT_INT64: return masm_->GetConstant<int64>(-1, repeat);
        default: return nullptr;
      }
    case REDUCE_OR:
      return nullptr;
    default:
      return nullptr;
  }
}

bool SIMDGenerator::SupportsUnroll() {
  return true;
}

void SIMDGenerator::Broadcast(int dst, const Operand &src) {
  // Broadcast is just a load for scalars.
  CHECK_EQ(VectorSize(), 1);
  Load(dst, src);
}

void SIMDGenerator::Sum(int r) {
  // Sum is a no-op for scalars.
  CHECK_EQ(VectorSize(), 1);
}

bool SIMDGenerator::SupportsMasking() {
  return false;
}

void SIMDGenerator::LoadNeutral(Reduction op, int r) {
  LOG(FATAL) << "Reduction not supported";
}

void SIMDGenerator::Accumulate(Reduction op, int acc, const jit::Operand &src) {
  LOG(FATAL) << "Reduction not supported";
}

void SIMDGenerator::Reduce(Reduction op, int r) {
  // Reduction is a no-op for scalars.
  CHECK_EQ(VectorSize(), 1);
}

void SIMDGenerator::SetMask(int bits) {
}

void SIMDGenerator::MaskedLoad(int dst, const Operand &src) {
  LOG(FATAL) << "Masking not supported";
}

void SIMDGenerator::MaskedStore(const Operand &dst, int src) {
  LOG(FATAL) << "Masking not supported";
}

void SIMDGenerator::MaskedAdd(int dst, int src1, const jit::Operand &src2) {
  LOG(FATAL) << "Masking not supported";
}

void SIMDGenerator::MaskedMul(int dst, int src1, const jit::Operand &src2) {
  LOG(FATAL) << "Masking not supported";
}

void SIMDGenerator::MaskedMulAdd(int dst, int src1, const jit::Operand &src2) {
  LOG(FATAL) << "Masking not supported";
}

void SIMDGenerator::MaskedAccumulate(Reduction op, int acc,
                                     const jit::Operand &src) {
  LOG(FATAL) << "Masking not supported";
}

// AVX512 float SIMD generator using 512-bit ZMM registers.
class AVX512FloatGenerator : public SIMDGenerator {
 public:
  AVX512FloatGenerator(MacroAssembler *masm, bool aligned)
      : SIMDGenerator(masm, aligned) {
    mask_ = masm->kk().alloc();
  }
  ~AVX512FloatGenerator() override {
    masm_->kk().release(mask_);
  }

  // Sixteen 32-bit floats per YMM register.
  int VectorBytes() override { return 64; }
  int VectorSize() override { return 16; }
  int Alloc() override { return masm_->mm().alloc(true); }

  void Load(int dst, const Operand &src) override {
    if (aligned_) {
      masm_->vmovaps(zmm(dst), src);
    } else {
      masm_->vmovups(zmm(dst), src);
    }
  }

  void Store(const Operand &dst, int src) override {
    if (aligned_) {
      masm_->vmovaps(dst, zmm(src));
    } else {
      masm_->vmovups(dst, zmm(src));
    }
  }

  void Broadcast(int dst, const Operand &src) override {
    masm_->vbroadcastss(zmm(dst), src);
  }

  void Zero(int r) override {
    masm_->vxorps(zmm(r), zmm(r), zmm(r));
  }

  void Add(int dst, int src1, int src2) override {
    masm_->vaddps(zmm(dst), zmm(src1), zmm(src2));
  }

  void Add(int dst, int src1, const jit::Operand &src2) override {
    masm_->vaddps(zmm(dst), zmm(src1), src2);
  }

  void Mul(int dst, int src1, const jit::Operand &src2) override {
    masm_->vmulps(zmm(dst), zmm(src1), src2);
  }

  void MulAdd(int dst, int src1, const Operand &src2, bool retain) override {
    masm_->vfmadd231ps(zmm(dst), zmm(src1), src2);
  }

  void Sum(int r) override {
    ZMMRegister sum = ZMMRegister::from_code(r);
    ZMMRegister acc = masm_->mm().allocz();
    masm_->Reduce(REDUCE_ADD, DT_FLOAT, sum, acc);
    masm_->mm().release(acc);
  }

  void LoadNeutral(Reduction op, int r) {
    StaticData *neutral = NeutralElement(op, DT_FLOAT);
    if (neutral == nullptr) {
      Zero(r);
    } else {
      masm_->vbroadcastss(zmm(r), neutral->address());
    }
  }

  void Accumulate(Reduction op, int acc, const jit::Operand &src) {
    masm_->Accumulate(op, DT_FLOAT, zmm(acc), src);
  }

  void Reduce(Reduction op, int r) {
    ZMMRegister aux = masm_->mm().allocz();
    masm_->Reduce(op, DT_FLOAT, zmm(r), aux);
    masm_->mm().release(aux);
  }

  bool SupportsMasking() override {
    return true;
  }

  void SetMask(int bits) override {
    masm_->LoadMask(bits, mask_);
  }

  void MaskedLoad(int dst, const jit::Operand &src) override {
    if (aligned_) {
      masm_->vmovaps(zmm(dst), src, Mask(mask_, zeroing));
    } else {
      masm_->vmovups(zmm(dst), src, Mask(mask_, zeroing));
    }
  }

  void MaskedStore(const jit::Operand &dst, int src) override  {
    if (aligned_) {
      masm_->vmovaps(dst, zmm(src), Mask(mask_, merging));
    } else {
      masm_->vmovups(dst, zmm(src), Mask(mask_, merging));
    }
  }

  void MaskedAdd(int dst, int src1, const jit::Operand &src2) override {
    masm_->vaddps(zmm(dst), zmm(src1), src2, Mask(mask_, merging));
  }

  void MaskedMul(int dst, int src1, const jit::Operand &src2) override {
    masm_->vmulps(zmm(dst), zmm(src1), src2, Mask(mask_, merging));
  }

  void MaskedMulAdd(int dst, int src1, const jit::Operand &src2) override {
    masm_->vfmadd231ps(zmm(dst), zmm(src1), src2, Mask(mask_, merging));
  }

  void MaskedAccumulate(Reduction op, int acc, const jit::Operand &src) {
    masm_->Accumulate(op, DT_FLOAT, zmm(acc), src, mask_);
  }

 private:
   OpmaskRegister mask_;
};

// AVX256 float SIMD generator using 256-bit YMM registers.
class AVX256FloatGenerator : public SIMDGenerator {
 public:
  AVX256FloatGenerator(MacroAssembler *masm, bool aligned)
      : SIMDGenerator(masm, aligned) {}

  // Eight 32-bit floats per YMM register.
  int VectorBytes() override { return 32; }
  int VectorSize() override { return 8; }
  int Alloc() override { return masm_->mm().alloc(false); }

  void Load(int dst, const Operand &src) override {
    if (aligned_) {
      masm_->vmovaps(ymm(dst), src);
    } else {
      masm_->vmovups(ymm(dst), src);
    }
  }

  void Store(const Operand &dst, int src) override {
    if (aligned_) {
      masm_->vmovaps(dst, ymm(src));
    } else {
      masm_->vmovups(dst, ymm(src));
    }
  }

  void Broadcast(int dst, const Operand &src) override {
    masm_->vbroadcastss(ymm(dst), src);
  }

  void Zero(int r) override {
    masm_->vxorps(ymm(r), ymm(r), ymm(r));
  }

  void Add(int dst, int src1, int src2) override {
    masm_->vaddps(ymm(dst), ymm(src1), ymm(src2));
  }

  void Add(int dst, int src1, const jit::Operand &src2) override {
    masm_->vaddps(ymm(dst), ymm(src1), src2);
  }

  void Mul(int dst, int src1, const jit::Operand &src2) override {
    masm_->vmulps(ymm(dst), ymm(src1), src2);
  }

  void MulAdd(int dst, int src1, const Operand &src2, bool retain) override {
    if (masm_->Enabled(FMA3)) {
      masm_->vfmadd231ps(ymm(dst), ymm(src1), src2);
    } else if (retain) {
      YMMRegister acc = masm_->mm().allocy();
      masm_->vmulps(acc, ymm(src1), src2);
      masm_->vaddps(ymm(dst), ymm(dst), acc);
      masm_->mm().release(acc);
    } else {
      masm_->vmulps(ymm(src1), ymm(src1), src2);
      masm_->vaddps(ymm(dst), ymm(dst), ymm(src1));
    }
  }

  void Sum(int r) override {
    YMMRegister sum = YMMRegister::from_code(r);
    YMMRegister acc = masm_->mm().allocy();
    masm_->Reduce(REDUCE_ADD, DT_FLOAT, sum, acc);
    masm_->mm().release(acc);
  }

  void LoadNeutral(Reduction op, int r) {
    StaticData *neutral = NeutralElement(op, DT_FLOAT);
    if (neutral == nullptr) {
      Zero(r);
    } else {
      masm_->vbroadcastss(ymm(r), neutral->address());
    }
  }

  void Accumulate(Reduction op, int acc, const jit::Operand &src) {
    masm_->Accumulate(op, DT_FLOAT, ymm(acc), src);
  }

  void Reduce(Reduction op, int r) {
    YMMRegister aux = masm_->mm().allocy();
    masm_->Reduce(op, DT_FLOAT, ymm(r), aux);
    masm_->mm().release(aux);
  }
};

// AVX128 float SIMD generator using 128-bit XMM registers.
class AVX128FloatGenerator : public SIMDGenerator {
 public:
  AVX128FloatGenerator(MacroAssembler *masm, bool aligned)
      : SIMDGenerator(masm, aligned) {}

  // Four 32-bit floats per XMM register.
  int VectorBytes() override { return 16; }
  int VectorSize() override { return 4; }
  int Alloc() override { return masm_->mm().alloc(false); }

  void Load(int dst, const Operand &src) override {
    if (aligned_) {
      masm_->vmovaps(xmm(dst), src);
    } else {
      masm_->vmovups(xmm(dst), src);
    }
  }

  void Store(const Operand &dst, int src) override {
    if (aligned_) {
      masm_->vmovaps(dst, xmm(src));
    } else {
      masm_->vmovups(dst, xmm(src));
    }
  }

  void Broadcast(int dst, const Operand &src) override {
    masm_->vbroadcastss(xmm(dst), src);
  }

  void Zero(int r) override {
    masm_->vxorps(xmm(r), xmm(r), xmm(r));
  }

  void Add(int dst, int src1, int src2) override {
    masm_->vaddps(xmm(dst), xmm(src1), xmm(src2));
  }

  void Add(int dst, int src1, const jit::Operand &src2) override {
    masm_->vaddps(xmm(dst), xmm(src1), src2);
  }

  void Mul(int dst, int src1, const jit::Operand &src2) override {
    masm_->vmulps(xmm(dst), xmm(src1), src2);
  }

  void MulAdd(int dst, int src1, const Operand &src2, bool retain) override {
    if (masm_->Enabled(FMA3)) {
      masm_->vfmadd231ps(xmm(dst), xmm(src1), src2);
    } else if (retain) {
      XMMRegister acc = masm_->mm().allocx();
      masm_->vmulps(acc, xmm(src1), src2);
      masm_->vaddps(xmm(dst), xmm(dst), acc);
      masm_->mm().release(acc);
    } else {
      masm_->vmulps(xmm(src1), xmm(src1), src2);
      masm_->vaddps(xmm(dst), xmm(dst), xmm(src1));
    }
  }

  void Sum(int r) override {
    XMMRegister sum = XMMRegister::from_code(r);
    XMMRegister acc = masm_->mm().allocx();
    masm_->Reduce(REDUCE_ADD, DT_FLOAT, sum, acc);
    masm_->mm().release(acc);
  }

  void LoadNeutral(Reduction op, int r) {
    StaticData *neutral = NeutralElement(op, DT_FLOAT);
    if (neutral == nullptr) {
      Zero(r);
    } else {
      masm_->vbroadcastss(xmm(r), neutral->address());
    }
  }

  void Accumulate(Reduction op, int acc, const jit::Operand &src) {
    masm_->Accumulate(op, DT_FLOAT, xmm(acc), src);
  }

  void Reduce(Reduction op, int r) {
    XMMRegister aux = masm_->mm().allocx();
    masm_->Reduce(op, DT_FLOAT, xmm(r), aux);
    masm_->mm().release(aux);
  }
};

// SSE128 float SIMD generator using 128-bit XMM registers.
class SSE128FloatGenerator : public SIMDGenerator {
 public:
  SSE128FloatGenerator(MacroAssembler *masm, bool aligned)
      : SIMDGenerator(masm, aligned) {}

  // Four 32-bit floats per YMM register.
  int VectorBytes() override { return 16; }
  int VectorSize() override { return 4; }
  int Alloc() override { return masm_->mm().alloc(false); }

  void Load(int dst, const Operand &src) override {
    if (aligned_) {
      masm_->movaps(xmm(dst), src);
    } else {
      masm_->movups(xmm(dst), src);
    }
  }

  void Store(const Operand &dst, int src) override {
    if (aligned_) {
      masm_->movaps(dst, xmm(src));
    } else {
      masm_->movups(dst, xmm(src));
    }
  }

  void Broadcast(int dst, const Operand &src) override {
    masm_->movss(xmm(dst), src);
    masm_->shufps(xmm(dst), xmm(dst), 0);
  }

  void Zero(int r) override {
    masm_->xorps(xmm(r), xmm(r));
  }

  void Add(int dst, int src1, int src2) override {
    if (dst != src1) masm_->movaps(xmm(dst), xmm(src1));
    masm_->addps(xmm(dst), xmm(src2));
  }

  void Add(int dst, int src1, const jit::Operand &src2) override {
    if (dst != src1) masm_->movaps(xmm(dst), xmm(src1));
    if (aligned_) {
      masm_->addps(xmm(dst), src2);
    } else {
      XMMRegister mem = masm_->mm().allocx();
      masm_->movups(mem, src2);
      masm_->addps(xmm(dst), mem);
      masm_->mm().release(mem);
    }
  }

  void Mul(int dst, int src1, const jit::Operand &src2) override {
    if (dst != src1) masm_->movaps(xmm(dst), xmm(src1));
    if (aligned_) {
      masm_->mulps(xmm(dst), src2);
    } else {
      XMMRegister mem = masm_->mm().allocx();
      masm_->movups(mem, src2);
      masm_->mulps(xmm(dst), mem);
      masm_->mm().release(mem);
    }
  }

  void MulAdd(int dst, int src1, const Operand &src2, bool retain) override {
    if (retain) {
      if (aligned_) {
        XMMRegister acc = masm_->mm().allocx();
        masm_->movaps(acc, xmm(src1));
        masm_->mulps(acc, src2);
        masm_->addps(xmm(dst), acc);
        masm_->mm().release(acc);
      } else {
        XMMRegister acc = masm_->mm().allocx();
        XMMRegister mem = masm_->mm().allocx();
        masm_->movaps(acc, xmm(src1));
        masm_->movups(mem, src2);
        masm_->mulps(acc, mem);
        masm_->addps(xmm(dst), acc);
        masm_->mm().release(acc);
        masm_->mm().release(mem);
      }
    } else {
      if (aligned_) {
        masm_->mulps(xmm(src1), src2);
        masm_->addps(xmm(dst), xmm(src1));
      } else {
        XMMRegister mem = masm_->mm().allocx();
        masm_->movups(mem, src2);
        masm_->mulps(xmm(src1), mem);
        masm_->addps(xmm(dst), xmm(src1));
        masm_->mm().release(mem);
      }
    }
  }

  void Sum(int r) override {
    XMMRegister sum = XMMRegister::from_code(r);
    XMMRegister acc = masm_->mm().allocx();
    masm_->Reduce(REDUCE_ADD, DT_FLOAT, sum, acc);
    masm_->mm().release(acc);
  }

  void LoadNeutral(Reduction op, int r) {
    StaticData *neutral = NeutralElement(op, DT_FLOAT, VectorSize());
    if (neutral == nullptr) {
      Zero(r);
    } else {
      Load(r, neutral->address());
    }
  }

  void Accumulate(Reduction op, int acc, const jit::Operand &src) {
    masm_->Accumulate(op, DT_FLOAT, xmm(acc), src);
  }

  void Reduce(Reduction op, int r) {
    XMMRegister aux = masm_->mm().allocx();
    masm_->Reduce(op, DT_FLOAT, xmm(r), aux);
    masm_->mm().release(aux);
  }
};

// AVX512 scalar float SIMD generator.
class AVX512ScalarFloatGenerator : public SIMDGenerator {
 public:
  AVX512ScalarFloatGenerator(MacroAssembler *masm, bool aligned)
      : SIMDGenerator(masm, aligned) {
    mask_ = masm->kk().alloc();
  }
  ~AVX512ScalarFloatGenerator() override {
    masm_->kk().release(mask_);
  }

  // Only uses the lower 32-bit float of ZMM register.
  int VectorBytes() override { return sizeof(float); }
  int VectorSize() override { return 1; }
  int Alloc() override { return masm_->mm().alloc(true); }

  void Load(int dst, const Operand &src) override {
    masm_->vmovss(zmm(dst), src);
  }

  void Store(const Operand &dst, int src) override {
    masm_->vmovss(dst, zmm(src));
  }

  void Zero(int r) override {
    masm_->vxorps(zmm(r), zmm(r), zmm(r));
  }

  void Add(int dst, int src1, int src2) override {
    masm_->vaddss(zmm(dst), zmm(src1), zmm(src2));
  }

  void Add(int dst, int src1, const jit::Operand &src2) override {
    masm_->vaddss(zmm(dst), zmm(src1), src2);
  }

  void Mul(int dst, int src1, const jit::Operand &src2) override {
    masm_->vmulss(zmm(dst), zmm(src1), src2);
  }

  void MulAdd(int dst, int src1, const Operand &src2, bool retain) override {
    if (masm_->Enabled(FMA3)) {
      masm_->vfmadd231ss(zmm(dst), zmm(src1), src2);
    } else if (retain) {
      ZMMRegister acc = masm_->mm().allocz();
      masm_->vmulss(acc, zmm(src1), src2);
      masm_->vaddss(zmm(dst), zmm(dst), acc);
      masm_->mm().release(acc);
    } else {
      masm_->vmulss(zmm(src1), zmm(src1), src2);
      masm_->vaddss(zmm(dst), zmm(dst), zmm(src1));
    }
  }

  void SetMask(int bits) override {
    masm_->LoadMask(bits, mask_);
  }

  void LoadNeutral(Reduction op, int r) {
    StaticData *neutral = NeutralElement(op, DT_FLOAT);
    if (neutral == nullptr) {
      Zero(r);
    } else {
      Load(r, neutral->address());
    }
  }

  void Accumulate(Reduction op, int acc, const jit::Operand &src) {
    masm_->Accumulate(op, DT_FLOAT, zmm(acc), src, mask_);
  }

 private:
  OpmaskRegister mask_;
};

// AVX scalar float SIMD generator.
class AVXScalarFloatGenerator : public SIMDGenerator {
 public:
  AVXScalarFloatGenerator(MacroAssembler *masm, bool aligned)
      : SIMDGenerator(masm, aligned) {}

  // Only uses the lower 32-bit float of XMM register.
  int VectorBytes() override { return sizeof(float); }
  int VectorSize() override { return 1; }
  int Alloc() override { return masm_->mm().alloc(false); }

  void Load(int dst, const Operand &src) override {
    masm_->vmovss(xmm(dst), src);
  }

  void Store(const Operand &dst, int src) override {
    masm_->vmovss(dst, xmm(src));
  }

  void Zero(int r) override {
    masm_->vxorps(xmm(r), xmm(r), xmm(r));
  }

  void Add(int dst, int src1, int src2) override {
    masm_->vaddss(xmm(dst), xmm(src1), xmm(src2));
  }

  void Add(int dst, int src1, const jit::Operand &src2) override {
    masm_->vaddss(xmm(dst), xmm(src1), src2);
  }

  void Mul(int dst, int src1, const jit::Operand &src2) override {
    masm_->vmulss(xmm(dst), xmm(src1), src2);
  }

  void MulAdd(int dst, int src1, const Operand &src2, bool retain) override {
    if (masm_->Enabled(FMA3)) {
      masm_->vfmadd231ss(xmm(dst), xmm(src1), src2);
    } else if (retain) {
      XMMRegister acc = masm_->mm().allocx();
      masm_->vmulss(acc, xmm(src1), src2);
      masm_->vaddss(xmm(dst), xmm(dst), acc);
      masm_->mm().release(acc);
    } else {
      masm_->vmulss(xmm(src1), xmm(src1), src2);
      masm_->vaddss(xmm(dst), xmm(dst), xmm(src1));
    }
  }

  void LoadNeutral(Reduction op, int r) {
    StaticData *neutral = NeutralElement(op, DT_FLOAT);
    if (neutral == nullptr) {
      Zero(r);
    } else {
      Load(r, neutral->address());
    }
  }

  void Accumulate(Reduction op, int acc, const jit::Operand &src) {
    switch (op) {
      case REDUCE_ADD:
        masm_->vaddss(xmm(acc), xmm(acc), src);
        break;
      case REDUCE_MUL:
        masm_->vmulss(xmm(acc), xmm(acc), src);
        break;
      case REDUCE_MIN:
        masm_->vminss(xmm(acc), xmm(acc), src);
        break;
      case REDUCE_MAX:
        masm_->vmaxss(xmm(acc), xmm(acc), src);
        break;
      case REDUCE_AND: {
        XMMRegister aux = masm_->mm().allocx();
        masm_->vmovss(aux, src);
        masm_->vandps(xmm(acc), xmm(acc), aux);
        masm_->mm().release(aux);
        break;
      }
      case REDUCE_OR: {
        XMMRegister aux = masm_->mm().allocx();
        masm_->vmovss(aux, src);
        masm_->vorps(xmm(acc), xmm(acc), aux);
        masm_->mm().release(aux);
        break;
      }
    }
  }
};

// SSE scalar float SIMD generator.
class SSEScalarFloatGenerator : public SIMDGenerator {
 public:
  SSEScalarFloatGenerator(MacroAssembler *masm, bool aligned)
      : SIMDGenerator(masm, aligned) {}

  // Only uses the lower 32-bit float of XMM register.
  int VectorBytes() override { return sizeof(float); }
  int VectorSize() override { return 1; }
  int Alloc() override { return masm_->mm().alloc(false); }

  void Load(int dst, const Operand &src) override {
    masm_->movss(xmm(dst), src);
  }

  void Store(const Operand &dst, int src) override {
    masm_->movss(dst, xmm(src));
  }

  void Zero(int r) override {
    masm_->xorps(xmm(r), xmm(r));
  }

  void Add(int dst, int src1, int src2) override {
    if (dst != src1) masm_->movss(xmm(dst), xmm(src1));
    masm_->addss(xmm(dst), xmm(src2));
  }

  void Add(int dst, int src1, const jit::Operand &src2) override {
    if (dst != src1) masm_->movss(xmm(dst), xmm(src1));
    masm_->addss(xmm(dst), src2);
  }

  void Mul(int dst, int src1, const jit::Operand &src2) override {
    if (dst != src1) masm_->movss(xmm(dst), xmm(src1));
    masm_->mulss(xmm(dst), src2);
  }

  void MulAdd(int dst, int src1, const Operand &src2, bool retain) override {
    if (retain) {
      XMMRegister acc = masm_->mm().allocx();
      masm_->movss(acc, xmm(src1));
      masm_->mulss(acc, src2);
      masm_->addss(xmm(dst), acc);
      masm_->mm().release(acc);
    } else {
      masm_->mulss(xmm(src1), src2);
      masm_->addss(xmm(dst), xmm(src1));
    }
  }

  void LoadNeutral(Reduction op, int r) {
    StaticData *neutral = NeutralElement(op, DT_FLOAT);
    if (neutral == nullptr) {
      Zero(r);
    } else {
      Load(r, neutral->address());
    }
  }

  void Accumulate(Reduction op, int acc, const jit::Operand &src) {
    switch (op) {
      case REDUCE_ADD:
        masm_->addss(xmm(acc), src);
        break;
      case REDUCE_MUL:
        masm_->mulss(xmm(acc), src);
        break;
      case REDUCE_MIN:
        masm_->minss(xmm(acc), src);
        break;
      case REDUCE_MAX:
        masm_->maxss(xmm(acc), src);
        break;
      case REDUCE_AND: {
        XMMRegister aux = masm_->mm().allocx();
        masm_->movss(aux, src);
        masm_->andps(xmm(acc), aux);
        masm_->mm().release(aux);
        break;
      }
      case REDUCE_OR: {
        XMMRegister aux = masm_->mm().allocx();
        masm_->movss(aux, src);
        masm_->orps(xmm(acc), aux);
        masm_->mm().release(aux);
        break;
      }
    }
  }
};

// AVX512 double SIMD generator using 512-bit ZMM registers.
class AVX512DoubleGenerator : public SIMDGenerator {
 public:
  AVX512DoubleGenerator(MacroAssembler *masm, bool aligned)
      : SIMDGenerator(masm, aligned) {
    mask_ = masm->kk().alloc();
  }
  ~AVX512DoubleGenerator() override {
    masm_->kk().release(mask_);
  }

  // Eight 64-bit floats per YMM register.
  int VectorBytes() override { return 64; }
  int VectorSize() override { return 8; }
  int Alloc() override { return masm_->mm().alloc(true); }

  void Load(int dst, const Operand &src) override {
    if (aligned_) {
      masm_->vmovapd(zmm(dst), src);
    } else {
      masm_->vmovupd(zmm(dst), src);
    }
  }

  void Store(const Operand &dst, int src) override {
    if (aligned_) {
      masm_->vmovapd(dst, zmm(src));
    } else {
      masm_->vmovupd(dst, zmm(src));
    }
  }

  void Broadcast(int dst, const Operand &src) override {
    masm_->vbroadcastsd(zmm(dst), src);
  }

  void Zero(int r) override {
    masm_->vxorpd(zmm(r), zmm(r), zmm(r));
  }

  void Add(int dst, int src1, int src2) override {
    masm_->vaddpd(zmm(dst), zmm(src1), zmm(src2));
  }

  void Add(int dst, int src1, const jit::Operand &src2) override {
    masm_->vaddpd(zmm(dst), zmm(src1), src2);
  }

  void Mul(int dst, int src1, const jit::Operand &src2) override {
    masm_->vmulpd(zmm(dst), zmm(src1), src2);
  }

  void MulAdd(int dst, int src1, const Operand &src2, bool retain) override {
    masm_->vfmadd231pd(zmm(dst), zmm(src1), src2);
  }

  void Sum(int r) override {
    ZMMRegister sum = ZMMRegister::from_code(r);
    ZMMRegister acc = masm_->mm().allocz();
    masm_->Reduce(REDUCE_ADD, DT_DOUBLE, sum, acc);
    masm_->mm().release(acc);
  }

  void LoadNeutral(Reduction op, int r) {
    StaticData *neutral = NeutralElement(op, DT_DOUBLE);
    if (neutral == nullptr) {
      Zero(r);
    } else {
      masm_->vbroadcastsd(zmm(r), neutral->address());
    }
  }

  void Accumulate(Reduction op, int acc, const jit::Operand &src) {
    masm_->Accumulate(op, DT_DOUBLE, zmm(acc), src);
  }

  void Reduce(Reduction op, int r) {
    ZMMRegister aux = masm_->mm().allocz();
    masm_->Reduce(op, DT_DOUBLE, zmm(r), aux);
    masm_->mm().release(aux);
  }

  bool SupportsMasking() override {
    return true;
  }

  void SetMask(int bits) override {
    masm_->LoadMask(bits, mask_);
  }

  void MaskedLoad(int dst, const jit::Operand &src) override {
    if (aligned_) {
      masm_->vmovapd(zmm(dst), src, Mask(mask_, zeroing));
    } else {
      masm_->vmovupd(zmm(dst), src, Mask(mask_, zeroing));
    }
  }

  void MaskedStore(const jit::Operand &dst, int src) override  {
    if (aligned_) {
      masm_->vmovapd(dst, zmm(src), Mask(mask_, merging));
    } else {
      masm_->vmovupd(dst, zmm(src), Mask(mask_, merging));
    }
  }

  void MaskedAdd(int dst, int src1, const jit::Operand &src2) override {
    masm_->vaddpd(zmm(dst), zmm(src1), src2, Mask(mask_, merging));
  }

  void MaskedMul(int dst, int src1, const jit::Operand &src2) override {
    masm_->vmulpd(zmm(dst), zmm(src1), src2, Mask(mask_, merging));
  }

  void MaskedMulAdd(int dst, int src1, const jit::Operand &src2) override {
    masm_->vfmadd231pd(zmm(dst), zmm(src1), src2, Mask(mask_, merging));
  }

  void MaskedAccumulate(Reduction op, int acc, const jit::Operand &src) {
    masm_->Accumulate(op, DT_DOUBLE, zmm(acc), src, mask_);
  }

 private:
   OpmaskRegister mask_;
};

// AVX256 double SIMD generator using 256-bit YMM registers.
class AVX256DoubleGenerator : public SIMDGenerator {
 public:
  AVX256DoubleGenerator(MacroAssembler *masm, bool aligned)
      : SIMDGenerator(masm, aligned) {}

  // Four 64-bit floats per YMM register.
  int VectorBytes() override { return 32; }
  int VectorSize() override { return 4; }
  int Alloc() override { return masm_->mm().alloc(false); }

  void Load(int dst, const Operand &src) override {
    if (aligned_) {
      masm_->vmovapd(ymm(dst), src);
    } else {
      masm_->vmovupd(ymm(dst), src);
    }
  }

  void Store(const Operand &dst, int src) override {
    if (aligned_) {
      masm_->vmovapd(dst, ymm(src));
    } else {
      masm_->vmovupd(dst, ymm(src));
    }
  }

  void Broadcast(int dst, const Operand &src) override {
    masm_->vbroadcastsd(ymm(dst), src);
  }

  void Zero(int r) override {
    masm_->vxorpd(ymm(r), ymm(r), ymm(r));
  }

  void Add(int dst, int src1, int src2) override {
    masm_->vaddpd(ymm(dst), ymm(src1), ymm(src2));
  }

  void Add(int dst, int src1, const jit::Operand &src2) override {
    masm_->vaddpd(ymm(dst), ymm(src1), src2);
  }

  void Mul(int dst, int src1, const jit::Operand &src2) override {
    masm_->vmulpd(ymm(dst), ymm(src1), src2);
  }

  void MulAdd(int dst, int src1, const Operand &src2, bool retain) override {
    if (masm_->Enabled(FMA3)) {
      masm_->vfmadd231pd(ymm(dst), ymm(src1), src2);
    } else if (retain) {
      YMMRegister acc = masm_->mm().allocy();
      masm_->vmulpd(acc, ymm(src1), src2);
      masm_->vaddpd(ymm(dst), ymm(dst), acc);
      masm_->mm().release(acc);
    } else {
      masm_->vmulpd(ymm(src1), ymm(src1), src2);
      masm_->vaddpd(ymm(dst), ymm(dst), ymm(src1));
    }
  }

  void Sum(int r) override {
    YMMRegister sum = YMMRegister::from_code(r);
    YMMRegister acc = masm_->mm().allocy();
    masm_->Reduce(REDUCE_ADD, DT_DOUBLE, sum, acc);
    masm_->mm().release(acc);
  }

  void LoadNeutral(Reduction op, int r) {
    StaticData *neutral = NeutralElement(op, DT_DOUBLE);
    if (neutral == nullptr) {
      Zero(r);
    } else {
      masm_->vbroadcastsd(ymm(r), neutral->address());
    }
  }

  void Accumulate(Reduction op, int acc, const jit::Operand &src) {
    masm_->Accumulate(op, DT_DOUBLE, ymm(acc), src);
  }

  void Reduce(Reduction op, int r) {
    YMMRegister aux = masm_->mm().allocy();
    masm_->Reduce(op, DT_DOUBLE, ymm(r), aux);
    masm_->mm().release(aux);
  }
};

// AVX128 double SIMD generator using 128-bit XMM registers.
class AVX128DoubleGenerator : public SIMDGenerator {
 public:
  AVX128DoubleGenerator(MacroAssembler *masm, bool aligned)
      : SIMDGenerator(masm, aligned) {}

  // Two 64-bit floats per XMM register.
  int VectorBytes() override { return 16; }
  int VectorSize() override { return 2; }
  int Alloc() override { return masm_->mm().alloc(false); }

  void Load(int dst, const Operand &src) override {
    if (aligned_) {
      masm_->vmovapd(xmm(dst), src);
    } else {
      masm_->vmovupd(xmm(dst), src);
    }
  }

  void Store(const Operand &dst, int src) override {
    if (aligned_) {
      masm_->vmovapd(dst, xmm(src));
    } else {
      masm_->vmovupd(dst, xmm(src));
    }
  }

  void Broadcast(int dst, const Operand &src) override {
    masm_->vmovsd(xmm(dst), src);
    masm_->vshufpd(xmm(dst), xmm(dst), xmm(dst), 0);
  }

  void Zero(int r) override {
    masm_->vxorpd(xmm(r), xmm(r), xmm(r));
  }

  void Add(int dst, int src1, int src2) override {
    masm_->vaddpd(xmm(dst), xmm(src1), xmm(src2));
  }

  void Add(int dst, int src1, const jit::Operand &src2) override {
    masm_->vaddpd(xmm(dst), xmm(src1), src2);
  }

  void Mul(int dst, int src1, const jit::Operand &src2) override {
    masm_->vmulpd(xmm(dst), xmm(src1), src2);
  }

  void MulAdd(int dst, int src1, const Operand &src2, bool retain) override {
    if (masm_->Enabled(FMA3)) {
      masm_->vfmadd231pd(xmm(dst), xmm(src1), src2);
    } else if (retain) {
      XMMRegister acc = masm_->mm().allocx();
      masm_->vmulpd(acc, xmm(src1), src2);
      masm_->vaddpd(xmm(dst), xmm(dst), acc);
      masm_->mm().release(acc);
    } else {
      masm_->vmulpd(xmm(src1), xmm(src1), src2);
      masm_->vaddpd(xmm(dst), xmm(dst), xmm(src1));
    }
  }

  void Sum(int r) override {
    XMMRegister sum = XMMRegister::from_code(r);
    XMMRegister acc = masm_->mm().allocx();
    masm_->Reduce(REDUCE_ADD, DT_DOUBLE, sum, acc);
    masm_->mm().release(acc);
  }

  void LoadNeutral(Reduction op, int r) {
    StaticData *neutral = NeutralElement(op, DT_DOUBLE, VectorSize());
    if (neutral == nullptr) {
      Zero(r);
    } else {
      Load(r, neutral->address());
    }
  }

  void Accumulate(Reduction op, int acc, const jit::Operand &src) {
    masm_->Accumulate(op, DT_DOUBLE, xmm(acc), src);
  }

  void Reduce(Reduction op, int r) {
    XMMRegister aux = masm_->mm().allocx();
    masm_->Reduce(op, DT_DOUBLE, xmm(r), aux);
    masm_->mm().release(aux);
  }
};

// SSE128 double SIMD generator using 128-bit XMM registers.
class SSE128DoubleGenerator : public SIMDGenerator {
 public:
  SSE128DoubleGenerator(MacroAssembler *masm, bool aligned)
      : SIMDGenerator(masm, aligned) {}

  // Two 32-bit floats per YMM register.
  int VectorBytes() override { return 16; }
  int VectorSize() override { return 2; }
  int Alloc() override { return masm_->mm().alloc(false); }

  void Load(int dst, const Operand &src) override {
    if (aligned_) {
      masm_->movapd(xmm(dst), src);
    } else {
      masm_->movupd(xmm(dst), src);
    }
  }

  void Store(const Operand &dst, int src) override {
    if (aligned_) {
      masm_->movapd(dst, xmm(src));
    } else {
      masm_->movupd(dst, xmm(src));
    }
  }

  void Broadcast(int dst, const Operand &src) override {
    masm_->movsd(xmm(dst), src);
    masm_->shufpd(xmm(dst), xmm(dst), 0);
  }

  void Zero(int r) override {
    masm_->xorpd(xmm(r), xmm(r));
  }

  void Add(int dst, int src1, int src2) override {
    if (dst != src1) masm_->movapd(xmm(dst), xmm(src1));
    masm_->addpd(xmm(dst), xmm(src2));
  }

  void Add(int dst, int src1, const jit::Operand &src2) override {
    if (dst != src1) masm_->movapd(xmm(dst), xmm(src1));
    if (aligned_) {
      masm_->addpd(xmm(dst), src2);
    } else {
      XMMRegister mem = masm_->mm().allocx();
      masm_->movupd(mem, src2);
      masm_->addpd(xmm(dst), mem);
      masm_->mm().release(mem);
    }
  }

  void Mul(int dst, int src1, const jit::Operand &src2) override {
    if (dst != src1) masm_->movapd(xmm(dst), xmm(src1));
    if (aligned_) {
      masm_->mulpd(xmm(dst), src2);
    } else {
      XMMRegister mem = masm_->mm().allocx();
      masm_->movupd(mem, src2);
      masm_->mulpd(xmm(dst), mem);
      masm_->mm().release(mem);
    }
  }

  void MulAdd(int dst, int src1, const Operand &src2, bool retain) override {
    if (retain) {
      if (aligned_) {
        XMMRegister acc = masm_->mm().allocx();
        masm_->movapd(acc, xmm(src1));
        masm_->mulpd(acc, src2);
        masm_->addpd(xmm(dst), acc);
        masm_->mm().release(acc);
      } else {
        XMMRegister acc = masm_->mm().allocx();
        XMMRegister mem = masm_->mm().allocx();
        masm_->movapd(acc, xmm(src1));
        masm_->movupd(mem, src2);
        masm_->mulpd(acc, mem);
        masm_->addpd(xmm(dst), acc);
        masm_->mm().release(acc);
        masm_->mm().release(mem);
      }
    } else {
      if (aligned_) {
        masm_->mulpd(xmm(src1), src2);
        masm_->addpd(xmm(dst), xmm(src1));
      } else {
        XMMRegister mem = masm_->mm().allocx();
        masm_->movupd(mem, src2);
        masm_->mulpd(xmm(src1), mem);
        masm_->addpd(xmm(dst), xmm(src1));
        masm_->mm().release(mem);
      }
    }
  }

  void Sum(int r) override {
    XMMRegister sum = XMMRegister::from_code(r);
    XMMRegister acc = masm_->mm().allocx();
    masm_->Reduce(REDUCE_ADD, DT_DOUBLE, sum, acc);
    masm_->mm().release(acc);
  }

  void LoadNeutral(Reduction op, int r) {
    StaticData *neutral = NeutralElement(op, DT_DOUBLE, VectorSize());
    if (neutral == nullptr) {
      Zero(r);
    } else {
      Load(r, neutral->address());
    }
  }

  void Accumulate(Reduction op, int acc, const jit::Operand &src) {
    masm_->Accumulate(op, DT_DOUBLE, xmm(acc), src);
  }

  void Reduce(Reduction op, int r) {
    XMMRegister aux = masm_->mm().allocx();
    masm_->Reduce(op, DT_DOUBLE, xmm(r), aux);
    masm_->mm().release(aux);
  }
};

// AVX512 scalar double SIMD generator.
class AVX512ScalarDoubleGenerator : public SIMDGenerator {
 public:
  AVX512ScalarDoubleGenerator(MacroAssembler *masm, bool aligned)
      : SIMDGenerator(masm, aligned) {
    mask_ = masm->kk().alloc();
  }
  ~AVX512ScalarDoubleGenerator() override {
    masm_->kk().release(mask_);
  }

  // Only uses the lower 64-bit float of ZMM register.
  int VectorBytes() override { return sizeof(double); }
  int VectorSize() override { return 1; }
  int Alloc() override { return masm_->mm().alloc(true); }

  void Load(int dst, const Operand &src) override {
    masm_->vmovsd(zmm(dst), src);
  }

  void Store(const Operand &dst, int src) override {
    masm_->vmovsd(dst, zmm(src));
  }

  void Zero(int r) override {
    masm_->vxorpd(zmm(r), zmm(r), zmm(r));
  }

  void Add(int dst, int src1, int src2) override {
    masm_->vaddsd(zmm(dst), zmm(src1), zmm(src2));
  }

  void Add(int dst, int src1, const jit::Operand &src2) override {
    masm_->vaddsd(zmm(dst), zmm(src1), src2);
  }

  void Mul(int dst, int src1, const jit::Operand &src2) override {
    masm_->vmulsd(zmm(dst), zmm(src1), src2);
  }

  void MulAdd(int dst, int src1, const Operand &src2, bool retain) override {
    if (masm_->Enabled(FMA3)) {
      masm_->vfmadd231sd(zmm(dst), zmm(src1), src2);
    } else if (retain) {
      ZMMRegister acc = masm_->mm().allocz();
      masm_->vmulsd(acc, zmm(src1), src2);
      masm_->vaddsd(zmm(dst), zmm(dst), acc);
      masm_->mm().release(acc);
    } else {
      masm_->vmulsd(zmm(src1), zmm(src1), src2);
      masm_->vaddsd(zmm(dst), zmm(dst), zmm(src1));
    }
  }

  void SetMask(int bits) override {
    masm_->LoadMask(bits, mask_);
  }

  void LoadNeutral(Reduction op, int r) {
    StaticData *neutral = NeutralElement(op, DT_DOUBLE);
    if (neutral == nullptr) {
      Zero(r);
    } else {
      Load(r, neutral->address());
    }
  }

  void Accumulate(Reduction op, int acc, const jit::Operand &src) {
    masm_->Accumulate(op, DT_DOUBLE, zmm(acc), src, mask_);
  }

 private:
  OpmaskRegister mask_;
};

// AVX scalar double SIMD generator.
class AVXScalarDoubleGenerator : public SIMDGenerator {
 public:
  AVXScalarDoubleGenerator(MacroAssembler *masm, bool aligned)
      : SIMDGenerator(masm, aligned) {}

  // Only uses the lower 64-bit float of XMM register.
  int VectorBytes() override { return sizeof(double); }
  int VectorSize() override { return 1; }
  int Alloc() override { return masm_->mm().alloc(false); }

  void Load(int dst, const Operand &src) override {
    masm_->vmovsd(xmm(dst), src);
  }

  void Store(const Operand &dst, int src) override {
    masm_->vmovsd(dst, xmm(src));
  }

  void Zero(int r) override {
    masm_->vxorpd(xmm(r), xmm(r), xmm(r));
  }

  void Add(int dst, int src1, int src2) override {
    masm_->vaddsd(xmm(dst), xmm(src1), xmm(src2));
  }

  void Add(int dst, int src1, const jit::Operand &src2) override {
    masm_->vaddsd(xmm(dst), xmm(src1), src2);
  }

  void Mul(int dst, int src1, const jit::Operand &src2) override {
    masm_->vmulsd(xmm(dst), xmm(src1), src2);
  }

  void MulAdd(int dst, int src1, const Operand &src2, bool retain) override {
    if (masm_->Enabled(FMA3)) {
      masm_->vfmadd231sd(xmm(dst), xmm(src1), src2);
    } else if (retain) {
      XMMRegister acc = masm_->mm().allocx();
      masm_->vmulsd(acc, xmm(src1), src2);
      masm_->vaddsd(xmm(dst), xmm(dst), acc);
      masm_->mm().release(acc);
    } else {
      masm_->vmulsd(xmm(src1), xmm(src1), src2);
      masm_->vaddsd(xmm(dst), xmm(dst), xmm(src1));
    }
  }

  void LoadNeutral(Reduction op, int r) {
    StaticData *neutral = NeutralElement(op, DT_DOUBLE);
    if (neutral == nullptr) {
      Zero(r);
    } else {
      Load(r, neutral->address());
    }
  }

  void Accumulate(Reduction op, int acc, const jit::Operand &src) {
    switch (op) {
      case REDUCE_ADD:
        masm_->vaddsd(xmm(acc), xmm(acc), src);
        break;
      case REDUCE_MUL:
        masm_->vmulsd(xmm(acc), xmm(acc), src);
        break;
      case REDUCE_MIN:
        masm_->vminsd(xmm(acc), xmm(acc), src);
        break;
      case REDUCE_MAX:
        masm_->vmaxsd(xmm(acc), xmm(acc), src);
        break;
      case REDUCE_AND: {
        XMMRegister aux = masm_->mm().allocx();
        masm_->vmovsd(aux, src);
        masm_->vandpd(xmm(acc), xmm(acc), aux);
        masm_->mm().release(aux);
        break;
      }
      case REDUCE_OR: {
        XMMRegister aux = masm_->mm().allocx();
        masm_->vmovsd(aux, src);
        masm_->vorpd(xmm(acc), xmm(acc), aux);
        masm_->mm().release(aux);
        break;
      }
    }
  }
};

// SSE scalar double SIMD generator.
class SSEScalarDoubleGenerator : public SIMDGenerator {
 public:
  SSEScalarDoubleGenerator(MacroAssembler *masm, bool aligned)
      : SIMDGenerator(masm, aligned) {}

  // Only uses the lower 64-bit float of XMM register.
  int VectorBytes() override { return sizeof(double); }
  int VectorSize() override { return 1; }
  int Alloc() override { return masm_->mm().alloc(false); }

  void Load(int dst, const Operand &src) override {
    masm_->movsd(xmm(dst), src);
  }

  void Store(const Operand &dst, int src) override {
    masm_->movsd(dst, xmm(src));
  }

  void Zero(int r) override {
    masm_->xorpd(xmm(r), xmm(r));
  }

  void Add(int dst, int src1, int src2) override {
    if (dst != src1) masm_->movsd(xmm(dst), xmm(src1));
    masm_->addsd(xmm(dst), xmm(src2));
  }

  void Add(int dst, int src1, const jit::Operand &src2) override {
    if (dst != src1) masm_->movsd(xmm(dst), xmm(src1));
    masm_->addsd(xmm(dst), src2);
  }

  void Mul(int dst, int src1, const jit::Operand &src2) override {
    if (dst != src1) masm_->movsd(xmm(dst), xmm(src1));
    masm_->mulsd(xmm(dst), src2);
  }

  void MulAdd(int dst, int src1, const Operand &src2, bool retain) override {
    if (retain) {
      XMMRegister acc = masm_->mm().allocx();
      masm_->movsd(acc, xmm(src1));
      masm_->mulsd(acc, src2);
      masm_->addsd(xmm(dst), acc);
      masm_->mm().release(acc);
    } else {
      masm_->mulsd(xmm(src1), src2);
      masm_->addsd(xmm(dst), xmm(src1));
    }
  }

  void LoadNeutral(Reduction op, int r) {
    StaticData *neutral = NeutralElement(op, DT_DOUBLE);
    if (neutral == nullptr) {
      Zero(r);
    } else {
      Load(r, neutral->address());
    }
  }

  void Accumulate(Reduction op, int acc, const jit::Operand &src) {
    switch (op) {
      case REDUCE_ADD:
        masm_->addsd(xmm(acc), src);
        break;
      case REDUCE_MUL:
        masm_->mulsd(xmm(acc), src);
        break;
      case REDUCE_MIN:
        masm_->minsd(xmm(acc), src);
        break;
      case REDUCE_MAX:
        masm_->maxsd(xmm(acc), src);
        break;
      case REDUCE_AND: {
        XMMRegister aux = masm_->mm().allocx();
        masm_->movsd(aux, src);
        masm_->andpd(xmm(acc), aux);
        masm_->mm().release(aux);
        break;
      }
      case REDUCE_OR: {
        XMMRegister aux = masm_->mm().allocx();
        masm_->movsd(aux, src);
        masm_->orpd(xmm(acc), aux);
        masm_->mm().release(aux);
        break;
      }
    }
  }
};

// Scalar integer SIMD generator using regular registers.
class ScalarIntSIMDGenerator : public SIMDGenerator {
 public:
  ScalarIntSIMDGenerator(MacroAssembler *masm, Type type)
      : SIMDGenerator(masm, false), type_(type) {}

  // Uses regular registers.
  int VectorBytes() override { return TypeTraits::of(type_).size(); }
  int VectorSize() override { return 1; }
  int Alloc() override { return masm_->rr().alloc().code(); }
  bool SupportsUnroll() override { return false; }

  void Load(int dst, const Operand &src) override {
    switch (type_) {
      case DT_INT8: masm_->movsxbq(reg(dst), src); break;
      case DT_INT16: masm_->movsxwq(reg(dst), src); break;
      case DT_INT32: masm_->movsxlq(reg(dst), src); break;
      case DT_INT64: masm_->movq(reg(dst), src); break;
      default: LOG(FATAL) << "Unsupported integer type: " << type_;
    }
  }

  void Store(const Operand &dst, int src) override {
    switch (type_) {
      case DT_INT8: masm_->movb(dst, reg(src)); break;
      case DT_INT16: masm_->movw(dst, reg(src)); break;
      case DT_INT32: masm_->movl(dst, reg(src)); break;
      case DT_INT64: masm_->movq(dst, reg(src)); break;
      default: LOG(FATAL) << "Unsupported integer type: " << type_;
    }
  }

  void Zero(int r) override {
    masm_->xorq(reg(r), reg(r));
  }

  void Add(int dst, int src1, int src2) override {
    if (dst == src1) {
      masm_->addq(reg(dst), reg(src2));
    } else if (dst == src2) {
      masm_->addq(reg(dst), reg(src1));
    } else {
      masm_->movq(reg(dst), reg(src1));
      masm_->addq(reg(dst), reg(src2));
    }
  }

  void Add(int dst, int src1, const jit::Operand &src2) override {
    if (dst == src1 && type_ == DT_INT64) {
      masm_->addq(reg(dst), src2);
    } else {
      Load(dst, src2);
      masm_->addq(reg(dst), reg(src1));
    }
  }

  void Mul(int dst, int src1, const jit::Operand &src2) override {
    if (dst == src1 && type_ == DT_INT64) {
      masm_->imulq(reg(dst), src2);
    } else {
      Load(dst, src2);
      masm_->imulq(reg(dst), reg(src1));
    }
  }

  void MulAdd(int dst, int src1, const Operand &src2, bool retain) override {
    if (!retain && type_ == DT_INT64) {
      masm_->imulq(reg(src1), src2);
      masm_->addq(reg(dst), reg(src1));
    } else {
      Register acc = masm_->rr().alloc();
      Load(acc.code(), src2);
      masm_->imulq(acc, reg(src1));
      masm_->addq(reg(dst), acc);
      masm_->rr().release(acc);
    }
  }

  void LoadNeutral(Reduction op, int r) {
    switch (op) {
      case REDUCE_ADD:
        Zero(r);
        break;
      case REDUCE_MUL:
        masm_->movq(reg(r), Immediate(1));
        break;
      case REDUCE_MIN:
        masm_->movq(reg(r), std::numeric_limits<int64>::max());
        break;
      case REDUCE_MAX:
        masm_->movq(reg(r), std::numeric_limits<int64>::min());
        break;
      case REDUCE_AND:
        masm_->movq(reg(r), Immediate(-1));
        break;
      case REDUCE_OR:
        Zero(r);
        break;
    }
  }

  void Accumulate(Reduction op, int acc, const jit::Operand &src) {
    switch (op) {
      case REDUCE_ADD:
        masm_->addq(reg(acc), src);
        break;
      case REDUCE_MUL:
        masm_->imulq(reg(acc), src);
        break;
      case REDUCE_MIN:
      case REDUCE_MAX: {
        Register aux = masm_->rr().try_alloc();
        if (aux.is_valid()) {
          Load(aux.code(), src);
          masm_->cmpq(reg(acc), aux);
          if (op == REDUCE_MIN) {
            masm_->cmovq(greater, reg(acc), aux);
          } else {
            masm_->cmovq(less, reg(acc), aux);
          }
          masm_->rr().release(acc);
        } else {
          switch (type_) {
            case DT_INT8:  masm_->cmpb(reg(acc), src); break;
            case DT_INT16: masm_->cmpw(reg(acc), src); break;
            case DT_INT32: masm_->cmpl(reg(acc), src); break;
            case DT_INT64: masm_->cmpq(reg(acc), src); break;
            default: LOG(FATAL) << "Unsupported integer type: " << type_;
          }
          if (op == REDUCE_MIN) {
            masm_->cmovq(greater, reg(acc), src);
          } else {
            masm_->cmovq(less, reg(acc), src);
          }
        }
        break;
      }
      case REDUCE_AND:
        masm_->andq(reg(acc), src);
        break;
      case REDUCE_OR:
        masm_->orq(reg(acc), src);
        break;
    }
  }

 public:
  Type type_;
};

bool SIMDAssembler::Supports(Type type) {
  return type == DT_FLOAT || type == DT_DOUBLE ||
         type == DT_INT8 || type == DT_INT16 ||
         type == DT_INT32 || type == DT_INT64;
}

int SIMDAssembler::RegisterUsage(Type type) {
  if (type == DT_INT8 || type == DT_INT16 ||
      type == DT_INT32 || type == DT_INT64) {
    return 1;
  } else {
    return 0;
  }
}

int SIMDAssembler::VectorBytes(Type type) {
  if (type == DT_FLOAT || type == DT_DOUBLE) {
    if (CPU::Enabled(AVX512F)) return 64;
    if (CPU::Enabled(AVX)) return 32;
    if (CPU::Enabled(SSE)) return 16;
  }
  return TypeTraits::of(type).size();
}

SIMDAssembler::SIMDAssembler(MacroAssembler *masm, Type type, bool aligned) {
  switch (type) {
    case DT_FLOAT:
      if (masm->Enabled(AVX512F)) {
        name_ = "AVX512Flt";
        add(new AVX512FloatGenerator(masm, aligned));
        add(new AVX512ScalarFloatGenerator(masm, aligned));
      } else if (masm->Enabled(AVX)) {
        name_ = "AVXFlt";
        add(new AVX256FloatGenerator(masm, aligned));
        add(new AVX128FloatGenerator(masm, aligned));
        add(new AVXScalarFloatGenerator(masm, aligned));
      } else if (masm->Enabled(SSE)) {
        name_ = "SSEFlt";
        add(new SSE128FloatGenerator(masm, aligned));
        add(new SSEScalarFloatGenerator(masm, aligned));
      }
      break;

    case DT_DOUBLE:
      if (masm->Enabled(AVX512F)) {
        name_ = "AVX512Dbl";
        add(new AVX512DoubleGenerator(masm, aligned));
        add(new AVX512ScalarDoubleGenerator(masm, aligned));
      } else if (masm->Enabled(AVX)) {
        name_ = "AVXDbl";
        add(new AVX256DoubleGenerator(masm, aligned));
        add(new AVX128DoubleGenerator(masm, aligned));
        add(new AVXScalarDoubleGenerator(masm, aligned));
      } else if (masm->Enabled(SSE)) {
        name_ = "SSEDbl";
        add(new SSE128DoubleGenerator(masm, aligned));
        add(new SSEScalarDoubleGenerator(masm, aligned));
      }
      break;

    case DT_INT8:
      name_ = "Int8";
      add(new ScalarIntSIMDGenerator(masm, DT_INT8));
      break;

    case DT_INT16:
      name_ = "Int16";
      add(new ScalarIntSIMDGenerator(masm, DT_INT16));
      break;

    case DT_INT32:
      name_ = "Int32";
      add(new ScalarIntSIMDGenerator(masm, DT_INT32));
      break;

    case DT_INT64:
      name_ = "Int64";
      add(new ScalarIntSIMDGenerator(masm, DT_INT64));
      break;

    default:
      LOG(FATAL) << "Unsuported type";
  }
}

SIMDAssembler::~SIMDAssembler() {
  for (auto *r : cascade_) delete r;
}

void SIMDAssembler::Sum(const std::vector<int> &regs) {
  if (regs.size() == 4) {
    main()->Add(regs[0], regs[0], regs[2]);
    main()->Add(regs[1], regs[1], regs[3]);
    main()->Add(regs[0], regs[0], regs[1]);
  } else {
    for (int n = 1; n < regs.size(); ++n) {
      main()->Add(regs[0], regs[0], regs[n]);
    }
  }
}

SIMDStrategy::SIMDStrategy(SIMDAssembler *sasm, int size) {
  // Use scalar generator for singletons.
  if (size == 1) {
    phases_.emplace_back(sasm->scalar());
    return;
  }

  // Add bulk phase.
  int vecsize = sasm->main()->VectorSize();
  int main = (size / vecsize) * vecsize;
  int max_unrolls = sasm->main()->SupportsUnroll() ? kMaxUnrolls : 1;
  int unrolls = std::min(main / vecsize, max_unrolls);
  int remaining = size;
  int offset = 0;
  if (unrolls > 0) {
    phases_.emplace_back(sasm->main());
    Phase &bulk = phases_.back();
    bulk.unrolls = unrolls;
    bulk.repeat = size / (vecsize * unrolls);
    remaining -= bulk.repeat * vecsize * unrolls;
    offset += bulk.repeat * vecsize * unrolls;
  }

  // Add residual phases.
  for (auto *gen : sasm->cascade()) {
    // Stop when all elements have been processed.
    if (remaining == 0) break;

    // Compute the number of elements that can be handled with this vector size.
    int vecsize = gen->VectorSize();
    int n = remaining / vecsize;
    if (n > 0) {
      // Add phase for generator.
      phases_.emplace_back(gen);
      Phase &phase = phases_.back();
      phase.unrolls = n;
      phase.offset = offset;
      offset += n * vecsize;
      remaining -= n * vecsize;
    }

    // Add masked phase for remainder if generator supports it.
    if (gen->SupportsMasking() && remaining > 0 && remaining < vecsize) {
      // Add masked phase for generator.
      phases_.emplace_back(gen);
      Phase &phase = phases_.back();
      phase.masked = remaining;
      phase.offset = offset;
      offset += remaining;
      remaining = 0;
    }
  }
}

int SIMDStrategy::MaxUnrolls() {
  int unrolls = 1;
  for (auto &p : phases_) unrolls = std::max(unrolls, p.unrolls);
  return unrolls;
}

void SIMDStrategy::PreloadMasks() {
  for (auto &p : phases_) {
    if (p.masked) {
      // Set mask for masked phase.
      p.generator->SetMask(p.masked);
    } else if (p.generator->VectorSize() == 1) {
      // Set singleton mask for scalar phase.
      p.generator->SetMask(1);
    }
  }
}

}  // namespace myelin
}  // namespace sling

