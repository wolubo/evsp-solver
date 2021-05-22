#include "CppUnitTest.h"
#include "EVSP.CudaSolver/CuVector2.hpp"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;


namespace EvspCudaSolverUnitTest
{
	TEST_CLASS(TestCuVector2)
	{
	public:

		TEST_METHOD(Test_CuVectorInt)
		{
			CuVector2<3> *v = new CuVector2<3>();
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
			CuVector2<3> *v = new CuVector2< 3>(42);
			for (int i = 0; i < 3; i++) {
				Assert::AreEqual(42, (*v)[i]);
			}
			delete v;
		}

		TEST_METHOD(Test_operatorEqual)
		{
			CuVector2<3> *a = new CuVector2< 3>();
			for (int i = 0; i < 3; i++) (*a)[i] = i;

			CuVector2<3> *b = new CuVector2< 3>();
			for (int i = 0; i < 3; i++) (*b)[i] = i;
			Assert::IsTrue((*a) == (*b));

			CuVector2<4> *c = new CuVector2< 4>(42);
			CuVector2<4> *d = new CuVector2<4>(42);
			Assert::IsTrue((*c) == (*d));

			delete a;
			delete b;
			delete c;
			delete d;
		}

		TEST_METHOD(Test_operatorNotEqual)
		{
			CuVector2<3> *a = new CuVector2< 3>();
			for (int i = 0; i < 3; i++) (*a)[i] = i;

			CuVector2<3> *b = new CuVector2< 3>();
			for (int i = 0; i < 3; i++) (*b)[i] = i;
			Assert::IsFalse((*a) != (*b));

			CuVector2<3> *c = new CuVector2< 3>(42);
			Assert::IsTrue((*a) != (*c));

			delete a;
			delete b;
			delete c;
		}

		TEST_METHOD(Test_setAll)
		{
			CuVector2<3> *a = new CuVector2< 3>();
			a->setAll(42);
			for (int i = 0; i < 3; i++) {
				Assert::AreEqual(42, (*a)[i]);
			}
			delete a;
		}

	};
}