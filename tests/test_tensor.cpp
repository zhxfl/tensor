#include "tensor.hpp"
#include <gtest/gtest.h>
using namespace tensor;
TEST(test_tensor, test_cuda){
    Tensor<float> t(-1, Size(3,2,2));
    for(size_t i = 0; i < t.size().count(); i++){
        t.mutable_cpu_data()[i] = i;
    }
    Tensor<float> t1(0, Size(3, 2,2));
    Tensor<float> t3(1, Size(3, 2, 2));
    t1.copy_from(t);
    t3.copy_from(t1);

    Tensor<float> t2(-1, Size(3, 2, 2));

    t2.copy_from(t3);
    for(size_t i = 0; i < t2.size().count(); i++){
        EXPECT_EQ(i, t2.mutable_cpu_data()[i]);
    }
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    return ret;
}

TEST(test_tensor, test_splice){
    Tensor<float> t(-1, Size(3,2,2));
    for(size_t i = 0; i < t.size().count(); i++){
        t.mutable_cpu_data()[i] = i;
    }

    Tensor<float>t1 = t.slice_from(Offset(0, 1, 1), Size(3, 1, 1));
    EXPECT_EQ(t1.mutable_cpu_data()[0], 9);
    EXPECT_EQ(t1.mutable_cpu_data()[1], 10);
    EXPECT_EQ(t1.mutable_cpu_data()[2], 11);
}

TEST(test_tensor, test_offset){
    Tensor<float> t(-1, Size(3,2,2));
    for(size_t i = 0; i < t.size().count(); i++){
        t.mutable_cpu_data()[i] = i;
    }

    EXPECT_EQ(t.mutable_cpu_data(Offset(0, 0, 0))[0], 0);
    EXPECT_EQ(t.mutable_cpu_data(Offset(1, 0, 0))[0], 1);
    EXPECT_EQ(t.mutable_cpu_data(Offset(2, 0, 0))[0], 2);
    EXPECT_EQ(t.mutable_cpu_data(Offset(0, 1, 0))[0], 3);
    EXPECT_EQ(t.mutable_cpu_data(Offset(1, 1, 0))[0], 4);
    EXPECT_EQ(t.mutable_cpu_data(Offset(2, 1, 0))[0], 5);
    EXPECT_EQ(t.mutable_cpu_data(Offset(0, 0, 1))[0], 6);
    EXPECT_EQ(t.mutable_cpu_data(Offset(1, 0, 1))[0], 7);
    EXPECT_EQ(t.mutable_cpu_data(Offset(2, 0, 1))[0], 8);
    EXPECT_EQ(t.mutable_cpu_data(Offset(0, 1, 1))[0], 9);
    EXPECT_EQ(t.mutable_cpu_data(Offset(1, 1, 1))[0], 10);
    EXPECT_EQ(t.mutable_cpu_data(Offset(2, 1, 1))[0], 11);

    Tensor<float>t1 = t.slice_from(Offset(0, 1, 1), Size(3, 1, 1));
    EXPECT_EQ(t1.mutable_cpu_data(Offset(0,0,0))[0], 9);
    EXPECT_EQ(t1.mutable_cpu_data(Offset(1,0,0))[0], 10);
    EXPECT_EQ(t1.mutable_cpu_data(Offset(2,0,0))[0], 11);


    Tensor<float>t2 = t1.slice_from(Offset(1, 0, 0), Size(2, 1, 1));
    EXPECT_EQ(t2.mutable_cpu_data(Offset(0,0,0))[0], 10);
    EXPECT_EQ(t2.mutable_cpu_data(Offset(1,0,0))[0], 11);
}
