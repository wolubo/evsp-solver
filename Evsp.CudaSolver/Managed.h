#pragma once

//#include <stdio.h>





class Managed
{
public:
	void* operator new(size_t len);
	void* operator new[](size_t len);
	void operator delete(void *ptr);
	void operator delete[](void *ptr);
};

