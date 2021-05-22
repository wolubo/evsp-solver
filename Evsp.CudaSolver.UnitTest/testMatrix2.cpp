#include "CppUnitTest.h"
#include "EVSP.CudaSolver/CuMatrix2.hpp"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;


namespace EvspCudaSolverUnitTest
{
	TEST_CLASS(TestMatrix2)
	{
	public:
		TEST_METHOD(Test_getNumOfX)
		{
			CuMatrix2<int,3,4> *matrix = new CuMatrix2<int,3,4>();

			delete matrix;
		}

		TEST_METHOD(Test_setAndGet)
		{
			CuMatrix2<int,3,4> *matrix = new CuMatrix2<int,3,4>();
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
			CuMatrix2<int,3,4> *matrix = new CuMatrix2<int,3,4>();
			for (int r = 0; r < 3; r++) {
				for (int c = 0; c < 4; c++) {
					int &i=matrix->itemAt(r, c);
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
			CuMatrix2<int,3,4> *matrix = new CuMatrix2<int,3,4>();
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
			CuMatrix2<int,3,4> *a = new CuMatrix2<int,3,4>();
			for (int r = 0; r < 3; r++) {
				for (int c = 0; c < 4; c++) {
					a->set(r, c, r*c + c);
				}
			}

			CuMatrix2<int,3,4> *b = new CuMatrix2<int,3,4>();
			for (int r = 0; r < 3; r++) {
				for (int c = 0; c < 4; c++) {
					b->set(r, c, r*c + c);
				}
			}

			Assert::IsTrue(*a==*b);

			CuMatrix2<int,3,4> *c = new CuMatrix2<int,3,4>();
			c->setAll(3);

			Assert::IsFalse(*a == *c);
			Assert::IsFalse(*b == *c);

			CuMatrix2<int,4,3> *d = new CuMatrix2<int,4,3>();
			d->setAll(3);

			CuMatrix2<int, 4, 3> *e = new CuMatrix2<int, 4, 3>();
			for (int r = 0; r < 4; r++) {
				for (int c = 0; c < 3; c++) {
					e->set(r, c, r*c + c);
				}
			}

			CuMatrix2<int,4, 3> *f = new CuMatrix2<int, 4, 3>();
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
			CuMatrix2<int,3,4> *a = new CuMatrix2<int,3,4>();
			for (int r = 0; r < 3; r++) {
				for (int c = 0; c < 4; c++) {
					a->set(r, c, r*c + c);
				}
			}

			CuMatrix2<int,3,4> *b = new CuMatrix2<int,3,4>();
			for (int r = 0; r < 3; r++) {
				for (int c = 0; c < 4; c++) {
					b->set(r, c, r*c + c);
				}
			}

			Assert::IsFalse(*a != *b);

			CuMatrix2<int,3,4> *c = new CuMatrix2<int,3,4>();
			c->setAll(3);

			Assert::IsTrue(*a != *c);
			Assert::IsTrue(*b != *c);

			CuMatrix2<int, 4, 3> *d = new CuMatrix2<int, 4, 3>();
			d->setAll(3);

			CuMatrix2<int, 4, 3> *e = new CuMatrix2<int, 4, 3>();
			for (int r = 0; r < 4; r++) {
				for (int c = 0; c < 3; c++) {
					e->set(r, c, r*c + c);
				}
			}

			CuMatrix2<int, 4, 3> *f = new CuMatrix2<int, 4, 3>();
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