#include "CppUnitTest.h"
#include "EVSP.CudaSolver/CuBoolVector1.hpp"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;


namespace EvspCudaSolverUnitTest
{
	TEST_CLASS(TestCuBinaryVector)
	{
	public:

		TEST_METHOD(Test_constructorAllFalse)
		{
			const unsigned int size = 200;
			CuBoolVector1 v(size, false);
			v.unsetAll();
			for (int i = 0; i < size; i++) {
				Assert::IsFalse(v.get(i));
			}
		}

		TEST_METHOD(Test_constructorAllTrue)
		{
			const unsigned int size = 200;
			CuBoolVector1 v(size, true);
			v.setAll();
			for (int i = 0; i < size; i++) {
				Assert::IsTrue(v.get(i));
			}
		}

		TEST_METHOD(Test_setAll)
		{
			const unsigned int size = 200;
			CuBoolVector1 v(size);
			v.setAll();
			for (int i = 0; i < size; i++) {
				Assert::IsTrue(v.get(i));
			}
		}

		TEST_METHOD(Test_unsetAll)
		{
			const unsigned int size = 200;
			CuBoolVector1 v(size);
			v.unsetAll();
			for (int i = 0; i < size; i++) {
				Assert::IsFalse(v.get(i));
			}
		}

		TEST_METHOD(Test_set_unset_toggle)
		{
			const unsigned int size = 10240;
			CuBoolVector1 v(size);
			v.unsetAll();
			for (int i = 0; i < size; i++) {
				if (i % 2 == 0)
					v.set(i);
				else
					v.unset(i);
			}

			for (int i = 0; i < size; i++) {
				if (i % 2 == 0)
					Assert::IsTrue(v.get(i));
				else
					Assert::IsFalse(v.get(i));
			}

			for (int i = 0; i < size; i++) {
				v.toggle(i);
			}

			for (int i = 0; i < size; i++) {
				if (i % 2 == 0)
					Assert::IsFalse(v.get(i));
				else
					Assert::IsTrue(v.get(i));
			}
		}


	};
}