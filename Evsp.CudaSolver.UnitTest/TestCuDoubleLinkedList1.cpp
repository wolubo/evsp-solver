#include "CppUnitTest.h"
#include "EVSP.CudaSolver/CuDoubleLinkedList1.hpp"
#include "EVSP.CudaSolver/CuSmartPtr.hpp"
#include <memory>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace EvspCudaSolverUnitTest
{

	//bool visitor(int &i, void* data1, const void *data2)
	//{
	//	CuDoubleLinkedList1<int> *result = (CuDoubleLinkedList1<int>*)data1;
	//	result->appendItem(i);
	//	return true;
	//}

	bool visitor(int &i, CuDoubleLinkedList1<int> *result, const int data2)
	{
		result->appendItem(i);
		return true;
	}

	TEST_CLASS(TestCuDoubleLinkedList1)
	{
	public:

		TEST_METHOD(Test_CuDoubleLinkedList1_Constructor)
		{
			try {
				std::shared_ptr<CuDoubleLinkedList1<int>> list(new CuDoubleLinkedList1<int>(2));
			}
			catch (...) {
				Assert::Fail();
			}

			try {
				std::shared_ptr<CuDoubleLinkedList1<int>> list(new CuDoubleLinkedList1<int>(0));
				Assert::Fail();
			}
			catch (...) {
			}
		}

		TEST_METHOD(Test_CuDoubleLinkedList1_isEmpty)
		{
			std::shared_ptr<CuDoubleLinkedList1<int>> list(new CuDoubleLinkedList1<int>(10));
			Assert::IsTrue(list->isEmpty());
			list->appendItem(1);
			Assert::IsFalse(list->isEmpty());
			list->appendItem(2);
			Assert::IsFalse(list->isEmpty());
			list->removeCurrentItem();
			Assert::IsFalse(list->isEmpty());
			list->removeCurrentItem();
			Assert::IsTrue(list->isEmpty());


		}

		TEST_METHOD(Test_CuDoubleLinkedList1_isFirst)
		{
			std::shared_ptr<CuDoubleLinkedList1<int>> list(new CuDoubleLinkedList1<int>(10));
			Assert::IsFalse(list->isFirst());
			list->appendItem(1);
			Assert::IsTrue(list->isFirst());
			list->appendItem(2);
			Assert::IsTrue(list->isFirst());
			int dummy;
			list->next(dummy);
			Assert::IsFalse(list->isFirst());


		}

		TEST_METHOD(Test_CuDoubleLinkedList1_isLast)
		{
			std::shared_ptr<CuDoubleLinkedList1<int>> list(new CuDoubleLinkedList1<int>(10));
			Assert::IsFalse(list->isLast());
			list->appendItem(1);
			Assert::IsTrue(list->isLast());
			list->appendItem(2);
			Assert::IsFalse(list->isLast());
			int dummy;
			list->next(dummy);
			Assert::IsTrue(list->isLast());


		}

		TEST_METHOD(Test_CuDoubleLinkedList1_hasPrev)
		{
			std::shared_ptr<CuDoubleLinkedList1<int>> list(new CuDoubleLinkedList1<int>(10));
			Assert::IsFalse(list->hasPrev());
			list->appendItem(1);
			Assert::IsFalse(list->hasPrev());
			list->appendItem(2);
			Assert::IsFalse(list->hasPrev());
			int dummy;
			list->next(dummy);
			Assert::IsTrue(list->hasPrev());


		}

		TEST_METHOD(Test_CuDoubleLinkedList1_hasNext)
		{
			std::shared_ptr<CuDoubleLinkedList1<int>> list(new CuDoubleLinkedList1<int>(10));
			Assert::IsFalse(list->hasNext());
			list->appendItem(1);
			Assert::IsFalse(list->hasNext());
			list->appendItem(2);
			Assert::IsTrue(list->hasNext());
			int dummy;
			list->next(dummy);
			Assert::IsFalse(list->hasNext());


		}

		TEST_METHOD(Test_CuDoubleLinkedList1_first)
		{
			int result = -1;
			std::shared_ptr<CuDoubleLinkedList1<int>> list(new CuDoubleLinkedList1<int>(10));
			result = list->first(result);
			Assert::AreEqual(-1, result);
			list->appendItem(1);
			result = list->first(result);
			Assert::AreEqual(1, result);
			list->appendItem(2);
			result = list->first(result);
			Assert::AreEqual(1, result);
			list->removeCurrentItem();
			result = list->first(result);
			Assert::AreEqual(2, result);


		}

		TEST_METHOD(Test_CuDoubleLinkedList1_last)
		{
			int result = -1;
			std::shared_ptr<CuDoubleLinkedList1<int>> list(new CuDoubleLinkedList1<int>(10));
			result = list->last(result);
			Assert::AreEqual(-1, result);
			list->appendItem(1);
			result = -1;
			result = list->last(result);
			Assert::AreEqual(1, result);
			list->appendItem(2);
			result = -1;
			result = list->last(result);
			Assert::AreEqual(2, result);
			list->removeCurrentItem();
			result = -1;
			result = list->last(result);
			Assert::AreEqual(1, result);


		}

		TEST_METHOD(Test_CuDoubleLinkedList1_prev)
		{
			int result = -1;
			std::shared_ptr<CuDoubleLinkedList1<int>> list(new CuDoubleLinkedList1<int>(10));
			result = list->prev(result);
			Assert::AreEqual(-1, result);
			list->appendItem(1);
			result = -1;
			result = list->prev(result);
			Assert::AreEqual(-1, result);
			list->appendItem(2);
			result = -1;
			result = list->prev(result);
			Assert::AreEqual(-1, result);
			result = list->last(result);
			result = -1;
			result = list->prev(result);
			Assert::AreEqual(1, result);
			list->removeCurrentItem();
			result = -1;
			result = list->prev(result);
			Assert::AreEqual(-1, result);


		}

		TEST_METHOD(Test_CuDoubleLinkedList1_curr)
		{
			std::shared_ptr<CuDoubleLinkedList1<int>> list(new CuDoubleLinkedList1<int>(10));
			int result = -1;
			result = list->curr(result);
			Assert::AreEqual(-1, result);

			list->appendItem(1);
			result = -1;
			result = list->curr(result);
			Assert::AreEqual(1, result);

			list->appendItem(2);
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


		}

		TEST_METHOD(Test_CuDoubleLinkedList1_next)
		{
			std::shared_ptr<CuDoubleLinkedList1<int>> list(new CuDoubleLinkedList1<int>(10));
			int result = -1;
			result = list->next(result);
			Assert::AreEqual(-1, result);
			list->appendItem(1);
			result = -1;
			result = list->next(result);
			Assert::AreEqual(-1, result);
			list->appendItem(2);
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


		}

		TEST_METHOD(Test_CuDoubleLinkedList1_insertBeforeCurrent)
		{
			std::shared_ptr<CuDoubleLinkedList1<int>> list(new CuDoubleLinkedList1<int>(10));

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


		}

		TEST_METHOD(Test_CuDoubleLinkedList1_insertAfterCurrent)
		{
			std::shared_ptr<CuDoubleLinkedList1<int>> list(new CuDoubleLinkedList1<int>(10));

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


		}

		TEST_METHOD(Test_CuDoubleLinkedList1_append)
		{
			std::shared_ptr<CuDoubleLinkedList1<int>> list(new CuDoubleLinkedList1<int>(10));

			int result = -1;
			list->appendItem(1);
			list->appendItem(2);
			list->appendItem(3);
			list->prev(result);
			list->appendItem(4);
			list->first(result);
			list->appendItem(5);

			result = -1;
			Assert::AreEqual(1, list->first(result));
			Assert::AreEqual(2, list->next(result));
			Assert::AreEqual(3, list->next(result));
			Assert::AreEqual(4, list->next(result));
			Assert::AreEqual(5, list->next(result));


		}

		TEST_METHOD(Test_CuDoubleLinkedList1_removeCurrent)
		{
			std::shared_ptr<CuDoubleLinkedList1<int>> list(new CuDoubleLinkedList1<int>(10));
			int result = -1;

			list->appendItem(1);
			list->appendItem(2);
			list->appendItem(3);
			list->appendItem(4);
			list->appendItem(5);

			list->first(result);
			Assert::AreEqual(5, (int)list->getSize());
			list->removeCurrentItem();
			Assert::AreEqual(4, (int)list->getSize());
			Assert::AreEqual(2, list->first(result));

			list->last(result);
			list->removeCurrentItem();
			Assert::AreEqual(3, (int)list->getSize());
			Assert::AreEqual(4, list->last(result));

			list->find(3);
			list->removeCurrentItem();
			Assert::AreEqual(2, (int)list->getSize());
			Assert::AreEqual(2, list->first(result));
			Assert::AreEqual(4, list->next(result));
			Assert::AreEqual(2, (int)list->getSize());


		}

		TEST_METHOD(Test_CuDoubleLinkedList1_find)
		{
			std::shared_ptr<CuDoubleLinkedList1<int>> list(new CuDoubleLinkedList1<int>(10));
			int result = -1;

			list->appendItem(1);
			list->appendItem(2);
			list->appendItem(3);
			list->appendItem(4);
			list->appendItem(5);

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


		}

		TEST_METHOD(Test_CuDoubleLinkedList1_copy1)
		{
			std::shared_ptr<CuDoubleLinkedList1<int>> list(new CuDoubleLinkedList1<int>(10));

			list->appendItem(1);
			list->appendItem(2);
			list->appendItem(3);
			list->appendItem(4);
			list->appendItem(5);

			shared_ptr<CuDoubleLinkedList1<int>> result = list->copy(ListRegion(ItemHandle::invalid(), ItemHandle::invalid(), false));
			Assert::AreEqual(5, (int)result->getSize());
			result->gotoFirst();
			Assert::AreEqual(1, result->curr());
			Assert::AreEqual(2, result->next());
			Assert::AreEqual(3, result->next());
			Assert::AreEqual(4, result->next());
			Assert::AreEqual(5, result->next());

			result = list->copy(ListRegion(ItemHandle::invalid(), ItemHandle::invalid(), true));
			Assert::AreEqual(5, (int)result->getSize());
			result->gotoFirst();
			Assert::AreEqual(1, result->curr());
			Assert::AreEqual(2, result->next());
			Assert::AreEqual(3, result->next());
			Assert::AreEqual(4, result->next());
			Assert::AreEqual(5, result->next());

			CuDoubleLinkedList1<int> result2(10);
			list->copy(ListRegion(ItemHandle::invalid(), ItemHandle::invalid(), false), result2);
			Assert::AreEqual(5, (int)result2.getSize());
			result2.gotoFirst();
			Assert::AreEqual(1, result2.curr());
			Assert::AreEqual(2, result2.next());
			Assert::AreEqual(3, result2.next());
			Assert::AreEqual(4, result2.next());
			Assert::AreEqual(5, result2.next());

			CuDoubleLinkedList1<int> result3(10);
			list->copy(ListRegion(ItemHandle::invalid(), ItemHandle::invalid(), true), result3);
			Assert::AreEqual(5, (int)result3.getSize());
			result3.gotoFirst();
			Assert::AreEqual(1, result3.curr());
			Assert::AreEqual(2, result3.next());
			Assert::AreEqual(3, result3.next());
			Assert::AreEqual(4, result3.next());
			Assert::AreEqual(5, result3.next());
		}

		TEST_METHOD(Test_CuDoubleLinkedList1_copy2)
		{
			std::shared_ptr<CuDoubleLinkedList1<int>> list(new CuDoubleLinkedList1<int>(10));

			list->appendItem(1);
			list->appendItem(2);
			list->appendItem(3);
			list->appendItem(4);
			list->appendItem(5);

			list->gotoFirst();
			ItemHandle firstItem = list->getCurrentPosition();

			list->gotoLast();
			ItemHandle lastItem = list->getCurrentPosition();

			shared_ptr<CuDoubleLinkedList1<int>> result = list->copy(ListRegion(firstItem, lastItem, true));
			Assert::AreEqual(5, (int)result->getSize());
			result->gotoFirst();
			Assert::AreEqual(1, result->curr());
			Assert::AreEqual(2, result->next());
			Assert::AreEqual(3, result->next());
			Assert::AreEqual(4, result->next());
			Assert::AreEqual(5, result->next());

			result = list->copy(ListRegion(firstItem, lastItem, false));
			Assert::AreEqual(3, (int)result->getSize());
			result->gotoFirst();
			Assert::AreEqual(2, result->curr());
			Assert::AreEqual(3, result->next());
			Assert::AreEqual(4, result->next());

			CuDoubleLinkedList1<int> result2(10);
			list->copy(ListRegion(firstItem, lastItem, true), result2);
			Assert::AreEqual(5, (int)result2.getSize());
			result2.gotoFirst();
			Assert::AreEqual(1, result2.curr());
			Assert::AreEqual(2, result2.next());
			Assert::AreEqual(3, result2.next());
			Assert::AreEqual(4, result2.next());
			Assert::AreEqual(5, result2.next());

			CuDoubleLinkedList1<int> result3(10);
			list->copy(ListRegion(firstItem, lastItem, false), result3);
			Assert::AreEqual(3, (int)result3.getSize());
			result3.gotoFirst();
			Assert::AreEqual(2, result3.curr());
			Assert::AreEqual(3, result3.next());
			Assert::AreEqual(4, result3.next());
		}

		TEST_METHOD(Test_CuDoubleLinkedList1_removeItems1)
		{
			std::shared_ptr<CuDoubleLinkedList1<int>> list(new CuDoubleLinkedList1<int>(10));

			{
				list->appendItem(1);
				list->appendItem(2);
				list->appendItem(3);
				list->appendItem(4);
				list->appendItem(5);

				list->removeItems();
				Assert::AreEqual(0, (int)list->getSize());
			}

			{
				list->appendItem(1);
				list->appendItem(2);
				list->appendItem(3);
				list->appendItem(4);
				list->appendItem(5);

				list->removeItems(ListRegion(ItemHandle::invalid(), ItemHandle::invalid(), false));
				Assert::AreEqual(0, (int)list->getSize());
			}

			{
				list->appendItem(1);
				list->appendItem(2);
				list->appendItem(3);
				list->appendItem(4);
				list->appendItem(5);

				list->removeItems(ListRegion(ItemHandle::invalid(), ItemHandle::invalid(), true));
				Assert::AreEqual(0, (int)list->getSize());
			}

			{
				list->appendItem(1);
				list->appendItem(2);
				list->appendItem(3);
				list->appendItem(4);
				list->appendItem(5);

				list->gotoFirst();
				ItemHandle firstItem = list->getCurrentPosition();

				list->gotoLast();
				ItemHandle lastItem = list->getCurrentPosition();

				list->removeItems(ListRegion(firstItem, lastItem, true));
				Assert::AreEqual(0, (int)list->getSize());
			}

			{
				list->appendItem(1);
				list->appendItem(2);
				list->appendItem(3);
				list->appendItem(4);
				list->appendItem(5);

				list->gotoFirst();
				ItemHandle firstItem = list->getCurrentPosition();

				list->gotoLast();
				ItemHandle lastItem = list->getCurrentPosition();

				list->removeItems(ListRegion(firstItem, lastItem, false));
				Assert::AreEqual(2, (int)list->getSize());
				list->gotoFirst();
				Assert::AreEqual(1, list->curr());
				Assert::AreEqual(5, list->next());
			}
		}

		TEST_METHOD(Test_CuDoubleLinkedList1_removeItems2)
		{
			std::shared_ptr<CuDoubleLinkedList1<int>> list(new CuDoubleLinkedList1<int>(10));

			list->appendItem(1);
			list->appendItem(2);
			list->appendItem(3);
			list->appendItem(4);
			list->appendItem(5);

			list->gotoFirst();
			list->gotoNext();
			ItemHandle itemToDelete = list->getCurrentPosition();

			list->removeItems(ListRegion(itemToDelete, itemToDelete, true));
			Assert::AreEqual(4, (int)list->getSize());
			list->gotoFirst();
			Assert::AreEqual(1, list->curr());
			Assert::AreEqual(3, list->next());
			Assert::AreEqual(4, list->next());
			Assert::AreEqual(5, list->next());
		}

		TEST_METHOD(Test_CuDoubleLinkedList1_visitItems1)
		{
			std::shared_ptr<CuDoubleLinkedList1<int>> list(new CuDoubleLinkedList1<int>(10));
			std::shared_ptr<CuDoubleLinkedList1<int>> result(new CuDoubleLinkedList1<int>(10));

			list->appendItem(1);
			list->appendItem(2);
			list->appendItem(3);
			list->appendItem(4);
			list->appendItem(5);

			list->visitItems<CuDoubleLinkedList1<int>*,int>(visitor, result.get(), 0);

			result->gotoFirst();
			Assert::AreEqual(1, result->curr());
			Assert::AreEqual(2, result->next());
			Assert::AreEqual(3, result->next());
			Assert::AreEqual(4, result->next());
			Assert::AreEqual(5, result->next());
		}

		TEST_METHOD(Test_CuDoubleLinkedList1_visitItems2)
		{
			std::shared_ptr<CuDoubleLinkedList1<int>> list(new CuDoubleLinkedList1<int>(10));
			std::shared_ptr<CuDoubleLinkedList1<int>> result(new CuDoubleLinkedList1<int>(10));

			list->appendItem(1);
			list->appendItem(2);
			list->appendItem(3);
			list->appendItem(4);
			list->appendItem(5);

			list->gotoFirst();
			ItemHandle left = list->getCurrentPosition();
			list->gotoLast();
			ItemHandle right = list->getCurrentPosition();

			ListRegion region(left, right, false);
			list->visitItems(visitor, result.get(), 0, region);

			Assert::AreEqual(3, (int)result->getSize());
			result->gotoFirst();
			Assert::AreEqual(2, result->curr());
			Assert::AreEqual(3, result->next());
			Assert::AreEqual(4, result->next());
		}


		TEST_METHOD(Test_CuDoubleLinkedList1_visitItems3)
		{
			std::shared_ptr<CuDoubleLinkedList1<int>> list(new CuDoubleLinkedList1<int>(10));
			std::shared_ptr<CuDoubleLinkedList1<int>> result(new CuDoubleLinkedList1<int>(10));

			list->appendItem(1);
			list->appendItem(2);
			list->appendItem(3);
			list->appendItem(4);
			list->appendItem(5);

			list->gotoFirst();
			ItemHandle left = list->getCurrentPosition();
			list->gotoLast();
			ItemHandle right = list->getCurrentPosition();

			ListRegion region(left, right, true);
			list->visitItems(visitor, result.get(), 0, region);

			result->gotoFirst();
			Assert::AreEqual(1, result->curr());
			Assert::AreEqual(2, result->next());
			Assert::AreEqual(3, result->next());
			Assert::AreEqual(4, result->next());
			Assert::AreEqual(5, result->next());
		}

		TEST_METHOD(Test_CuDoubleLinkedList1_visitItems4)
		{
			std::shared_ptr<CuDoubleLinkedList1<int>> list(new CuDoubleLinkedList1<int>(10));
			std::shared_ptr<CuDoubleLinkedList1<int>> result(new CuDoubleLinkedList1<int>(10));

			list->appendItem(1);
			list->appendItem(2);
			list->appendItem(3);
			list->appendItem(4);
			list->appendItem(5);

			list->gotoFirst();
			list->gotoNext();
			list->gotoNext();
			ItemHandle left = list->getCurrentPosition();
			ItemHandle right = list->getCurrentPosition();

			ListRegion region(left, right, true);
			list->visitItems(visitor, result.get(), 0, region);

			Assert::AreEqual(1, (int)result->getSize());
			result->gotoFirst();
			Assert::AreEqual(3, result->curr());
		}

		TEST_METHOD(Test_CuDoubleLinkedList1_visitItems5)
		{
			std::shared_ptr<CuDoubleLinkedList1<int>> list(new CuDoubleLinkedList1<int>(10));
			std::shared_ptr<CuDoubleLinkedList1<int>> result(new CuDoubleLinkedList1<int>(10));

			list->appendItem(1);
			list->appendItem(2);
			list->appendItem(3);
			list->appendItem(4);
			list->appendItem(5);

			list->gotoFirst();
			list->gotoNext();
			list->gotoNext();
			ItemHandle left;
			ItemHandle right = list->getCurrentPosition();

			ListRegion region(left, right, false);
			list->visitItems(visitor, result.get(), 0, region);

			Assert::AreEqual(2, (int)result->getSize());
			result->gotoFirst();
			Assert::AreEqual(1, result->curr());
			Assert::AreEqual(2, result->next());
		}

		TEST_METHOD(Test_CuDoubleLinkedList1_visitItems6)
		{
			std::shared_ptr<CuDoubleLinkedList1<int>> list(new CuDoubleLinkedList1<int>(10));
			std::shared_ptr<CuDoubleLinkedList1<int>> result(new CuDoubleLinkedList1<int>(10));

			list->appendItem(1);
			list->appendItem(2);
			list->appendItem(3);
			list->appendItem(4);
			list->appendItem(5);

			list->gotoFirst();
			list->gotoNext();
			list->gotoNext();
			ItemHandle left;
			ItemHandle right = list->getCurrentPosition();

			ListRegion region(left, right, true);
			list->visitItems(visitor, result.get(), 0, region);

			Assert::AreEqual(3, (int)result->getSize());
			result->gotoFirst();
			Assert::AreEqual(1, result->curr());
			Assert::AreEqual(2, result->next());
			Assert::AreEqual(3, result->next());
		}

		TEST_METHOD(Test_CuDoubleLinkedList1_visitItemsReversedOrder1)
		{
			std::shared_ptr<CuDoubleLinkedList1<int>> list(new CuDoubleLinkedList1<int>(10));
			std::shared_ptr<CuDoubleLinkedList1<int>> result(new CuDoubleLinkedList1<int>(10));

			list->appendItem(1);
			list->appendItem(2);
			list->appendItem(3);
			list->appendItem(4);
			list->appendItem(5);

			list->visitItemsReversedOrder(visitor, result.get(), 0);

			result->gotoFirst();
			Assert::AreEqual(5, result->curr());
			Assert::AreEqual(4, result->next());
			Assert::AreEqual(3, result->next());
			Assert::AreEqual(2, result->next());
			Assert::AreEqual(1, result->next());
		}

		TEST_METHOD(Test_CuDoubleLinkedList1_visitItemsReversedOrder2)
		{
			std::shared_ptr<CuDoubleLinkedList1<int>> list(new CuDoubleLinkedList1<int>(10));
			std::shared_ptr<CuDoubleLinkedList1<int>> result(new CuDoubleLinkedList1<int>(10));

			list->appendItem(1);
			list->appendItem(2);
			list->appendItem(3);
			list->appendItem(4);
			list->appendItem(5);

			list->gotoFirst();
			ItemHandle left = list->getCurrentPosition();
			list->gotoLast();
			ItemHandle right = list->getCurrentPosition();

			ListRegion region(left, right, false);
			list->visitItemsReversedOrder(visitor, result.get(), 0, region);

			Assert::AreEqual(3, (int)result->getSize());
			result->gotoFirst();
			Assert::AreEqual(4, result->curr());
			Assert::AreEqual(3, result->next());
			Assert::AreEqual(2, result->next());
		}


		TEST_METHOD(Test_CuDoubleLinkedList1_visitItemsReversedOrder3)
		{
			std::shared_ptr<CuDoubleLinkedList1<int>> list(new CuDoubleLinkedList1<int>(10));
			std::shared_ptr<CuDoubleLinkedList1<int>> result(new CuDoubleLinkedList1<int>(10));

			list->appendItem(1);
			list->appendItem(2);
			list->appendItem(3);
			list->appendItem(4);
			list->appendItem(5);

			list->gotoFirst();
			ItemHandle left = list->getCurrentPosition();
			list->gotoLast();
			ItemHandle right = list->getCurrentPosition();

			ListRegion region(left, right, true);
			list->visitItemsReversedOrder(visitor, result.get(), 0, region);

			result->gotoFirst();
			Assert::AreEqual(5, result->curr());
			Assert::AreEqual(4, result->next());
			Assert::AreEqual(3, result->next());
			Assert::AreEqual(2, result->next());
			Assert::AreEqual(1, result->next());
		}

		TEST_METHOD(Test_CuDoubleLinkedList1_appendList1)
		{
			std::shared_ptr<CuDoubleLinkedList1<int>> list1(new CuDoubleLinkedList1<int>(10));
			std::shared_ptr<CuDoubleLinkedList1<int>> list2(new CuDoubleLinkedList1<int>(10));

			list1->appendItem(1);
			list1->appendItem(2);
			list1->appendItem(3);
			list1->appendItem(4);
			list1->appendItem(5);

			list2->appendItem(6);
			list2->appendItem(7);
			list2->appendItem(8);
			list2->appendItem(9);

			list1->appendList(*list2);

			Assert::AreEqual(9, (int)list1->getSize());
			list1->gotoFirst();
			Assert::AreEqual(1, list1->curr());
			Assert::AreEqual(2, list1->next());
			Assert::AreEqual(3, list1->next());
			Assert::AreEqual(4, list1->next());
			Assert::AreEqual(5, list1->next());
			Assert::AreEqual(6, list1->next());
			Assert::AreEqual(7, list1->next());
			Assert::AreEqual(8, list1->next());
			Assert::AreEqual(9, list1->next());
		}	

		TEST_METHOD(Test_CuDoubleLinkedList1_appendList2)
		{
			std::shared_ptr<CuDoubleLinkedList1<int>> list1(new CuDoubleLinkedList1<int>(10));
			std::shared_ptr<CuDoubleLinkedList1<int>> list2(new CuDoubleLinkedList1<int>(10));

			list2->appendItem(1);
			list2->appendItem(2);
			list2->appendItem(3);
			list2->appendItem(4);
			list2->appendItem(5);

			list1->appendList(*list2);

			Assert::AreEqual(5, (int)list1->getSize());
			list1->gotoFirst();
			Assert::AreEqual(1, list1->curr());
			Assert::AreEqual(2, list1->next());
			Assert::AreEqual(3, list1->next());
			Assert::AreEqual(4, list1->next());
			Assert::AreEqual(5, list1->next());
		}
	};

}