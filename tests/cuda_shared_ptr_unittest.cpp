#include "../cuda_shared_ptr.hpp"
#include "test_object.h"
#include <gtest/gtest.h>

class CudaSharedPtrTest : public testing::Test
{
protected:
    CudaSharedPtrTest()
    {
        nullTestObject = nullptr;
    }

    TestObject* createSampleTestObject()
    {
        TestObject* d_sampleTestObject;
        TestObject* h_sampleTestObject = new TestObject(69);
        cudaMalloc((void**)&d_sampleTestObject, sizeof(TestObject));
        cudaMemcpy(
            d_sampleTestObject,
            h_sampleTestObject,
            sizeof(TestObject),
            cudaMemcpyHostToDevice);
        delete h_sampleTestObject;
        return d_sampleTestObject;
    }

    TestObject* nullTestObject;
};

TEST_F(CudaSharedPtrTest, DefaultConstructorNullPointer)
{
  cuda_shared_ptr<TestObject> ptr(nullptr);
  EXPECT_EQ(nullTestObject, ptr.get());
}

TEST_F(CudaSharedPtrTest, DefaultConstructorZero)
{
    size_t n = 0;
    cuda_shared_ptr<TestObject> ptr(n);
    EXPECT_EQ(nullTestObject, ptr.get());
}

TEST_F(CudaSharedPtrTest, PointerConstructorEqualsPointer)
{
  TestObject* to = createSampleTestObject();
  cuda_shared_ptr<TestObject> ptr(to);
  EXPECT_EQ(to, ptr.get());
}

TEST_F(CudaSharedPtrTest, ResetClearsPointer)
{
  cuda_shared_ptr<TestObject> ptr(createSampleTestObject());
  EXPECT_NE(nullTestObject, ptr.get());
  ptr.reset(nullptr);
  EXPECT_EQ(nullTestObject, ptr.get());
}
