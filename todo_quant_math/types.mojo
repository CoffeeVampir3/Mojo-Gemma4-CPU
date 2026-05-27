from std.memory import UnsafePointer


comptime F32Ptr = UnsafePointer[Float32, MutAnyOrigin]
comptime BF16Ptr = UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]
comptime I8Ptr = UnsafePointer[Scalar[DType.int8], MutAnyOrigin]
