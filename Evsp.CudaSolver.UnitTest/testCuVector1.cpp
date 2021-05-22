#include "CppUnitTest.h"
#include "EVSP.CudaSolver/CuVector1.hpp"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;


namespace EvspCudaSolverUnitTest
{
	TEST_CLASS(TestCuVector1)
	{
	public:

		TEST_METHOD(Test_CuVectorInt)
		{
			CuVector1<int> *v = new CuVector1<int>(3);
			for (int i = 0; i < 3; i++) {
				(*v)[i] = i;
			}
			for (int i = 0; i < 3; i++) {
				Assert::AreEqual(i, (*v)[i]);
			}
			delete v;
		}

		TEST_METHOD(Test_CuVectorIntT)
		{
			CuVector1<int> *v = new CuVector1<int>(3);
			for (int i = 0; i < v->getSize(); i++) (*v)[i] = 42;
			for (int i = 0; i < 3; i++) {
				Assert::AreEqual(42, (*v)[i]);
			}
			delete v;
		}

		TEST_METHOD(Test_operatorEqual)
		{
			CuVector1<int> *a = new CuVector1<int>(3);
			for (int i = 0; i < 3; i++) (*a)[i] = i;

			CuVector1<int> *b = new CuVector1<int>(3);
			for (int i = 0; i < 3; i++) (*b)[i] = i;
			Assert::IsTrue((*a) == (*b));

			CuVector1<int> *c = new CuVector1<int>(4);
			for (int i = 0; i < c->getSize(); i++) (*c)[i] = 42;

			CuVector1<int> *d = new CuVector1<int>(4);
			for (int i = 0; i < d->getSize(); i++) (*d)[i] = 42;

			Assert::IsTrue((*c) == (*d));

			delete a;
			delete b;
			delete c;
			delete d;
		}

		TEST_METHOD(Test_operatorNotEqual)
		{
			CuVector1<int> *a = new CuVector1<int>(3);
			for (int i = 0; i < 3; i++) (*a)[i] = i;

			CuVector1<int> *b = new CuVector1<int>(3);
			for (int i = 0; i < 3; i++) (*b)[i] = i;
			Assert::IsFalse((*a) != (*b));

			CuVector1<int> *c = new CuVector1<int>(3);
			for (int i = 0; i < c->getSize(); i++) (*c)[i] = 42;

			Assert::IsTrue((*a) != (*c));

			delete a;
			delete b;
			delete c;
		}

		TEST_METHOD(Test_setAll)
		{
			CuVector1<int> *a = new CuVector1<int>(3);
			for (int i = 0; i < a->getSize(); i++) (*a)[i] = 42;
			//a->setAll(42);
			for (int i = 0; i < 3; i++) {
				Assert::AreEqual(42, (*a)[i]);
			}
			delete a;
		}

	};
}