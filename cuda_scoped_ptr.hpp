/*
 * (C) Copyright Karol Dzitkowski 2015
 *
 */

#define CUDA_CALL(x) do { assert((x)==cudaSuccess); } while(0)

#ifndef CUDA_SMART_PTR_SCOPED_PTR_HPP_
#define CUDA_SMART_PTR_SCOPED_PTR_HPP_

#include <cuda_runtime_api.h>
#include <assert.h>
#include <cstddef>

template <class T> class cuda_scoped_ptr // noncopyable
{
private:
    T * ptr;

    cuda_scoped_ptr(cuda_scoped_ptr const &);
    cuda_scoped_ptr & operator=(cuda_scoped_ptr const &);

    void operator==(cuda_scoped_ptr const &) const;
    void operator!=(cuda_scoped_ptr const &) const;

public:
    explicit cuda_scoped_ptr()
    {
        CUDA_CALL( cudaMalloc((void**)&ptr, sizeof(T)) );
    }
    explicit cuda_scoped_ptr(size_t size)
    {
        if (size == 0) ptr = nullptr;
        else CUDA_CALL( cudaMalloc((void**)&ptr, size) );
    }
    explicit cuda_scoped_ptr(T * p = 0) : ptr(p) {}
    ~cuda_scoped_ptr()
    {
        CUDA_CALL( cudaFree(ptr) );
    }
    void reset(T * p = 0)
    {
        assert(p == 0 || p != ptr);
        cuda_scoped_ptr<T>(p).swap(*this);
    }
    T & operator*() const
    {
        assert(ptr != 0);
        return *ptr;
    }
    T * operator->() const
    {
        assert(ptr != 0);
        return ptr;
    }
    T * get() const { return ptr; }
    operator bool() const
    {
        return ptr != 0;
    }
    void swap(cuda_scoped_ptr & csp)
    {
        T * tmp = csp.ptr;
        csp.ptr = ptr;
        ptr = tmp;
    }
};

template<class T> inline bool operator==(
    cuda_scoped_ptr<T> const & p, std::nullptr_t) noexcept
{
    return p.get() == 0;
}

template<class T> inline bool operator==(
    std::nullptr_t, cuda_scoped_ptr<T> const & p) noexcept
{
    return p.get() == 0;
}

template<class T> inline bool operator!=(
    cuda_scoped_ptr<T> const & p, std::nullptr_t) noexcept
{
    return p.get() != 0;
}

template<class T> inline bool operator!=(
    std::nullptr_t, cuda_scoped_ptr<T> const & p) noexcept
{
    return p.get() != 0;
}

#endif
