#include "CppUnitTest.h"
#include "EVSP.CudaSolver/CuMatrix1.hpp"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;


namespace EvspCudaSolverUnitTest
{
	TEST_CLASS(TestMatrix1)
	{
	public:
		TEST_METHOD(Test_getNumOfX)
		{
			CuMatrix1<int> *matrix = new CuMatrix1<int>(3,4);
			Assert::AreEqual((int)3, matrix->getNumOfRows());
			Assert::AreEqual((int)4, matrix->getNumOfCols());
			delete matrix;
		}

		TEST_METHOD(Test_setAndGet)
		{
			CuMatrix1<int> *matrix = new CuMatrix1<int>(3, 4);
			for (int r = 0; r < 3; r++) {
				for (int c = 0; c < 4; c++) {
					matrix->set(r, c, r*c + c);
				}
			}
			for (int r = 0; r < 3; r++) {
				for (int c = 0; c < 4; c++) {
					Assert::AreEqual(r*c + c, matrix->get(r, c));
				}
			}
			delete matrix;
		}

		TEST_METHOD(Test_itemAt)
		{
			CuMatrix1<int> *matrix = new CuMatrix1<int>(3, 4);
			for (int r = 0; r < 3; r++) {
				for (int c = 0; c < 4; c++) {
					int &i = matrix->itemAt(r, c);
					i = r*c + c;
				}
			}
			for (int r = 0; r < 3; r++) {
				for (int c = 0; c < 4; c++) {
					Assert::AreEqual(r*c + c, matrix->get(r, c));
				}
			}
			delete matrix;
		}

		TEST_METHOD(Test_setAll)
		{
			CuMatrix1<int> *matrix = new CuMatrix1<int>(3,4);
			matrix->setAll(42);
			for (int r = 0; r < 3; r++) {
				for (int c = 0; c < 4; c++) {
					Assert::AreEqual(42, matrix->get(r, c));
				}
			}
			delete matrix;
		}

		TEST_METHOD(Test_operatorEqual)
		{
			CuMatrix1<int> *a = new CuMatrix1<int>(3,4);
			for (int r = 0; r < 3; r++) {
				for (int c = 0; c < 4; c++) {
					a->set(r, c, r*c + c);
				}
			}

			CuMatrix1<int> *b = new CuMatrix1<int>(3,4);
			for (int r = 0; r < 3; r++) {
				for (int c = 0; c < 4; c++) {
					b->set(r, c, r*c + c);
				}
			}

			Assert::IsTrue(*a == *b);

			CuMatrix1<int> *c = new CuMatrix1<int>(3,4);
			c->setAll(3);

			Assert::IsFalse(*a == *c);
			Assert::IsFalse(*b == *c);

			CuMatrix1<int> *d = new CuMatrix1<int>(4, 3);
			d->setAll(3);

			CuMatrix1<int> *e = new CuMatrix1<int>(4, 3);
			for (int r = 0; r < 4; r++) {
				for (int c = 0; c < 3; c++) {
					e->set(r, c, r*c + c);
				}
			}

			CuMatrix1<int> *f = new CuMatrix1<int>(4, 3);
			f->setAll(3);

			Assert::IsTrue(*d == *f);
			Assert::IsFalse(*e == *f);

			delete a;
			delete b;
			delete c;
			delete d;
			delete e;
			delete f;
		}

		TEST_METHOD(Test_operatorNotEqual)
		{
			CuMatrix1<int> *a = new CuMatrix1<int>(3,4);
			for (int r = 0; r < 3; r++) {
				for (int c = 0; c < 4; c++) {
					a->set(r, c, r*c + c);
				}
			}

			CuMatrix1<int> *b = new CuMatrix1<int>(3, 4);
			for (int r = 0; r < 3; r++) {
				for (int c = 0; c < 4; c++) {
					b->set(r, c, r*c + c);
				}
			}

			Assert::IsFalse(*a != *b);

			CuMatrix1<int> *c = new CuMatrix1<int>(3, 4);
			c->setAll(3);

			Assert::IsTrue(*a != *c);
			Assert::IsTrue(*b != *c);

			CuMatrix1<int> *d = new CuMatrix1<int>(4, 3);
			d->setAll(3);

			CuMatrix1<int> *e = new CuMatrix1<int>(4, 3);
			for (int r = 0; r < 4; r++) {
				for (int c = 0; c < 3; c++) {
					e->set(r, c, r*c + c);
				}
			}

			CuMatrix1<int> *f = new CuMatrix1<int>(4,3);
			f->setAll(3);

			Assert::IsFalse(*d != *f);
			Assert::IsTrue(*e != *f);

			delete a;
			delete b;
			delete c;
			delete d;
			delete e;
			delete f;
		}

	};
}