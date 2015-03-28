#ifndef TEST_OBJECT_H_
#define TEST_OBJECT_H_

class TestObject
{
private:
    int data;

public:
    TestObject(){}
    TestObject(int data) : data(data) {}
    ~TestObject(){}

    int TestMethod()
    {
        return data;
    }
};

#endif
