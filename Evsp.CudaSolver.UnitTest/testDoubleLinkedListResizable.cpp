#include "CppUnitTest.h"
#include "EVSP.CudaSolver/DoubleLinkedListResizable.hpp"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace EvspCudaSolverUnitTest
{
	TEST_CLASS(TestDoubleLinkedList)
	{
	public:

		TEST_METHOD(Test_isEmpty)
		{
			DoubleLinkedListResizable<int> *list = new DoubleLinkedListResizable<int>(10, 10);
			Assert::IsTrue(list->isEmpty());
			list->append(1);
			Assert::IsFalse(list->isEmpty());
			list->append(2);
			Assert::IsFalse(list->isEmpty());
			list->removeCurrent();
			Assert::IsFalse(list->isEmpty());
			list->removeCurrent();
			Assert::IsTrue(list->isEmpty());

			delete list;
		}

		TEST_METHOD(Test_isFirst)
		{
			DoubleLinkedListResizable<int> *list = new DoubleLinkedListResizable<int>(10, 10);
			Assert::IsFalse(list->isFirst());
			list->append(1);
			Assert::IsTrue(list->isFirst());
			list->append(2);
			Assert::IsTrue(list->isFirst());
			int dummy;
			list->next(dummy);
			Assert::IsFalse(list->isFirst());

			delete list;
		}

		TEST_METHOD(Test_isLast)
		{
			DoubleLinkedListResizable<int> *list = new DoubleLinkedListResizable<int>(10, 10);
			Assert::IsFalse(list->isLast());
			list->append(1);
			Assert::IsTrue(list->isLast());
			list->append(2);
			Assert::IsFalse(list->isLast());
			int dummy;
			list->next(dummy);
			Assert::IsTrue(list->isLast());

			delete list;
		}

		TEST_METHOD(Test_hasPrev)
		{
			DoubleLinkedListResizable<int> *list = new DoubleLinkedListResizable<int>(10, 10);
			Assert::IsFalse(list->hasPrev());
			list->append(1);
			Assert::IsFalse(list->hasPrev());
			list->append(2);
			Assert::IsFalse(list->hasPrev());
			int dummy;
			list->next(dummy);
			Assert::IsTrue(list->hasPrev());

			delete list;
		}

		TEST_METHOD(Test_hasNext)
		{
			DoubleLinkedListResizable<int> *list = new DoubleLinkedListResizable<int>(10, 10);
			Assert::IsFalse(list->hasNext());
			list->append(1);
			Assert::IsFalse(list->hasNext());
			list->append(2);
			Assert::IsTrue(list->hasNext());
			int dummy;
			list->next(dummy);
			Assert::IsFalse(list->hasNext());

			delete list;
		}

		TEST_METHOD(Test_first)
		{
			int result = -1;
			DoubleLinkedListResizable<int> *list = new DoubleLinkedListResizable<int>(10, 10);
			result = list->first(result);
			Assert::AreEqual(-1, result);
			list->append(1);
			result = list->first(result);
			Assert::AreEqual(1, result);
			list->append(2);
			result = list->first(result);
			Assert::AreEqual(1, result);
			list->removeCurrent();
			result = list->first(result);
			Assert::AreEqual(2, result);

			delete list;
		}

		TEST_METHOD(Test_last)
		{
			int result = -1;
			DoubleLinkedListResizable<int> *list = new DoubleLinkedListResizable<int>(10, 10);
			result = list->last(result);
			Assert::AreEqual(-1, result);
			list->append(1);
			result = -1;
			result = list->last(result);
			Assert::AreEqual(1, result);
			list->append(2);
			result = -1;
			result = list->last(result);
			Assert::AreEqual(2, result);
			list->removeCurrent();
			result = -1;
			result = list->last(result);
			Assert::AreEqual(1, result);

			delete list;
		}

		TEST_METHOD(Test_prev)
		{
			int result = -1;
			DoubleLinkedListResizable<int> *list = new DoubleLinkedListResizable<int>(10, 10);
			result = list->prev(result);
			Assert::AreEqual(-1, result);
			list->append(1);
			result = -1;
			result = list->prev(result);
			Assert::AreEqual(-1, result);
			list->append(2);
			result = -1;
			result = list->prev(result);
			Assert::AreEqual(-1, result);
			result = list->last(result);
			result = -1;
			result = list->prev(result);
			Assert::AreEqual(1, result);
			list->removeCurrent();
			result = -1;
			result = list->prev(result);
			Assert::AreEqual(-1, result);

			delete list;
		}

		TEST_METHOD(Test_curr)
		{
			DoubleLinkedListResizable<int> *list = new DoubleLinkedListResizable<int>(10, 10);
			int result = -1;
			result = list->curr(result);
			Assert::AreEqual(-1, result);

			list->append(1);
			result = -1;
			result = list->curr(result);
			Assert::AreEqual(1, result);

			list->append(2);
			result = -1;
			result = list->curr(result);
			Assert::AreEqual(1, result);

			list->last(result);
			result = -1;
			result = list->curr(result);
			Assert::AreEqual(2, result);

			list->first(result);
			result = -1;
			result = list->curr(result);
			Assert::AreEqual(1, result);

			delete list;
		}

		TEST_METHOD(Test_next)
		{
			DoubleLinkedListResizable<int> *list = new DoubleLinkedListResizable<int>(10, 10);
			int result = -1;
			result = list->next(result);
			Assert::AreEqual(-1, result);
			list->append(1);
			result = -1;
			result = list->next(result);
			Assert::AreEqual(-1, result);
			list->append(2);
			result = -1;
			result = list->next(result);
			Assert::AreEqual(2, result);
			result = list->first(result);
			result = -1;
			result = list->next(result);
			Assert::AreEqual(2, result);
			result = list->last(result);
			result = -1;
			result = list->next(result);
			Assert::AreEqual(-1, result);

			delete list;
		}

		TEST_METHOD(Test_insertBeforeCurrent)
		{
			DoubleLinkedListResizable<int> *list = new DoubleLinkedListResizable<int>(10, 10);

			int result = -1;
			list->insertItemBeforeCurrent(1);
			result = list->first(result);
			Assert::AreEqual(1, result);

			result = -1;
			list->insertItemBeforeCurrent(2);
			result = list->curr(result);
			Assert::AreEqual(1, result);
			result = -1;
			result = list->first(result);
			Assert::AreEqual(2, result);

			result = list->last(result);
			list->insertItemBeforeCurrent(3);
			result = -1;
			result = list->curr(result);
			Assert::AreEqual(1, result);
			result = -1;
			result = list->prev(result);
			Assert::AreEqual(3, result);

			result = list->first(result);
			result = list->next(result);
			list->insertItemBeforeCurrent(4);
			result = -1;
			result = list->curr(result);
			Assert::AreEqual(3, result);
			result = -1;
			result = list->prev(result);
			Assert::AreEqual(4, result);

			delete list;
		}

		TEST_METHOD(Test_insertAfterCurrent)
		{
			DoubleLinkedListResizable<int> *list = new DoubleLinkedListResizable<int>(10, 10);

			int result = -1;
			list->insertItemAfterCurrent(1);
			list->insertItemAfterCurrent(4);
			list->insertItemAfterCurrent(2);
			list->next(result);
			list->insertItemAfterCurrent(3);

			result = -1;
			result = list->first(result);
			Assert::AreEqual(1, result);
			result = list->next(result);
			Assert::AreEqual(2, result);
			result = list->next(result);
			Assert::AreEqual(3, result);
			result = list->next(result);
			Assert::AreEqual(4, result);

			delete list;
		}

		TEST_METHOD(Test_append)
		{
			DoubleLinkedListResizable<int> *list = new DoubleLinkedListResizable<int>(10, 10);

			int result = -1;
			list->append(1);
			list->append(2);
			list->append(3);
			list->prev(result);
			list->append(4);
			list->first(result);
			list->append(5);

			result = -1;
			Assert::AreEqual(1, list->first(result));
			Assert::AreEqual(2, list->next(result));
			Assert::AreEqual(3, list->next(result));
			Assert::AreEqual(4, list->next(result));
			Assert::AreEqual(5, list->next(result));

			delete list;
		}

		TEST_METHOD(Test_removeCurrent)
		{
			DoubleLinkedListResizable<int> *list = new DoubleLinkedListResizable<int>(10, 10);
			int result = -1;

			list->append(1);
			list->append(2);
			list->append(3);
			list->append(4);
			list->append(5);

			list->first(result);
			list->removeCurrent();
			Assert::AreEqual(2, list->first(result));

			list->last(result);
			list->removeCurrent();
			Assert::AreEqual(4, list->last(result));

			list->find(3);
			list->removeCurrent();
			Assert::AreEqual(2, list->first(result));
			Assert::AreEqual(4, list->next(result));
			Assert::AreEqual(2, (int)list->getSize());

			delete list;
		}

		TEST_METHOD(Test_find)
		{
			DoubleLinkedListResizable<int> *list = new DoubleLinkedListResizable<int>(10, 10);
			int result = -1;

			list->append(1);
			list->append(2);
			list->append(3);
			list->append(4);
			list->append(5);

			Assert::IsTrue(list->find(1));
			Assert::AreEqual(1, list->curr(result));

			Assert::IsTrue(list->find(3));
			Assert::AreEqual(3, list->curr(result));

			Assert::IsTrue(list->find(5));
			Assert::AreEqual(5, list->curr(result));

			Assert::IsTrue(list->find(2));
			Assert::AreEqual(2, list->curr(result));

			Assert::IsTrue(list->find(4));
			Assert::AreEqual(4, list->curr(result));

			Assert::IsTrue(list->find(1));
			Assert::AreEqual(1, list->curr(result));

			Assert::IsFalse(list->find(9));
			Assert::AreEqual(1, list->curr(result));

			delete list;
		}

		TEST_METHOD(Test_grow)
		{
			DoubleLinkedListResizable<int> *list = new DoubleLinkedListResizable<int>(10, 10);
			int result = -1;

			Assert::AreEqual(10, (int)list->getCapacity());
			list->append(1);
			list->append(2);
			list->append(3);
			list->append(4);
			list->append(5);
			list->append(6);
			list->append(7);
			list->append(8);
			list->append(9);
			list->append(10);
			list->append(11);
			Assert::AreEqual(11, (int)list->getSize());
			Assert::AreEqual(20, (int)list->getCapacity());

			result = -1;
			Assert::AreEqual(1, list->first(result));
			Assert::AreEqual(2, list->next(result));
			Assert::AreEqual(3, list->next(result));
			Assert::AreEqual(4, list->next(result));
			Assert::AreEqual(5, list->next(result));
			Assert::AreEqual(6, list->next(result));
			Assert::AreEqual(7, list->next(result));
			Assert::AreEqual(8, list->next(result));
			Assert::AreEqual(9, list->next(result));
			Assert::AreEqual(10, list->next(result));
			Assert::AreEqual(11, list->next(result));

			delete list;
		}

	};
}