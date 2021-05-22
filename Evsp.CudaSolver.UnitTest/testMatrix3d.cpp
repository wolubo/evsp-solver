#include "CppUnitTest.h"
#include "EVSP.CudaSolver/Matrix3d.hpp"
#include "EVSP.BaseClasses/Typedefs.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;


namespace EvspCudaSolverUnitTest
{
	TEST_CLASS(TestMatrix3d)
	{
	public:
		TEST_METHOD(Test_Matrix3dIntIntInt)
		{
			Matrix3d<int> *matrix = new Matrix3d<int>(3, 3, 3);
			Assert::AreEqual((size_t)3, matrix->getSizeX());
			Assert::AreEqual((size_t)3, matrix->getSizeY());
			Assert::AreEqual((size_t)3, matrix->getSizeZ());
			for (int x = 0; x < 3; x++) {
				for (int y = 0; y < 3; y++) {
					for (int z = 0; z < 3; z++) {
						matrix->set(x, y, z, x*y*z+x+y+z);
					}
				}
			}
			for (int x = 0; x < 3; x++) {
				for (int y = 0; y < 3; y++) {
					for (int z = 0; z < 3; z++) {
						Assert::AreEqual(x*y*z + x + y + z, matrix->get(x, y, z));
					}
				}
			}
			delete matrix;
		}

		TEST_METHOD(Test_Matrix3dIntIntIntInt)
		{
			Matrix3d<int> *matrix = new Matrix3d<int>(3, 3, 3, 42);
			Assert::AreEqual((size_t)3, matrix->getSizeX());
			Assert::AreEqual((size_t)3, matrix->getSizeY());
			Assert::AreEqual((size_t)3, matrix->getSizeZ());
			for (int x = 0; x < 3; x++) {
				for (int y = 0; y < 3; y++) {
					for (int z = 0; z < 3; z++) {
						Assert::AreEqual(42, matrix->get(x, y, z));
					}
				}
			}
			delete matrix;
		}


		TEST_METHOD(Test_itemAt)
		{
			Matrix3d<int> *matrix = new Matrix3d<int>(3, 3, 3);
			for (int x = 0; x < 3; x++) {
				for (int y = 0; y < 3; y++) {
					for (int z = 0; z < 3; z++) {
						int &i = matrix->itemAt(x, y, z);
						i = x*y*z + x + y + z;
					}
				}
			}
			for (int x = 0; x < 3; x++) {
				for (int y = 0; y < 3; y++) {
					for (int z = 0; z < 3; z++) {
						Assert::AreEqual(x*y*z + x + y + z, matrix->get(x, y, z));
					}
				}
			}
			delete matrix;
		}

		TEST_METHOD(Test_setAll)
		{
			Matrix3d<int> *matrix = new Matrix3d<int>(3, 3, 3);
			matrix->setAll(42);
			for (int x = 0; x < 3; x++) {
				for (int y = 0; y < 3; y++) {
					for (int z = 0; z < 3; z++) {
						Assert::AreEqual(42, matrix->get(x, y, z));
					}
				}
			}
			delete matrix;
		}

	};
}