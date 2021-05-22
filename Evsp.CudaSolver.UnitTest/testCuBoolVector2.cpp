#include "CppUnitTest.h"
#include "EVSP.CudaSolver/CuBoolVector2.hpp"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;


namespace EvspCudaSolverUnitTest
{
	TEST_CLASS(TestCuBinaryVector)
	{
	public:

		TEST_METHOD(Test_setAll)
		{
			const unsigned int size = 200;
			CuBoolVector2<unsigned int, size> v;
			v.setAll();
			for (int i = 0; i < size; i++) {
				Assert::IsTrue(v.get(i));
			}
		}

		TEST_METHOD(Test_unsetAll)
		{
			const unsigned int size = 200;
			CuBoolVector2<unsigned int, size> v;
			v.unsetAll();
			for (int i = 0; i < size; i++) {
				Assert::IsFalse(v.get(i));
			}
		}

		TEST_METHOD(Test_set_unset_toggle)
		{
			const unsigned int size = 10240;
			CuBoolVector2<unsigned int, size> v;
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