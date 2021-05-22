#pragma once

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda.h> 
#else
#include <stdexcept>
#endif

#include <assert.h>
#include <memory>
#include "EVSP.BaseClasses/DeviceLaunchParameters.h"
#include "EVSP.BaseClasses/Typedefs.h"
#include "CudaCheck.h"
#include "CuVector1.hpp"

using namespace std;

/// <summary>
/// Definiert einen Bereich innerhalb einer Liste.
/// </summary>
struct ListRegion {

	ListRegion(ItemHandle theLeft = ItemHandle::invalid(),
		ItemHandle theRight = ItemHandle::invalid(),
		bool theIncludeBorders = false)
		: left(theLeft), right(theRight), includeBorders(theIncludeBorders) {}

	/// <summary>
	/// Linke Grenze des Bereichs. Der Bereich beginnt mit dem Element rechts vom mit 'left' referenzierten Element. Das 
	/// referenzierte Element selbst gehört also nicht zum Bereich. Wenn 'left' ungültig ist, beginnt der Bereich mit dem 
	/// ersten Element der Liste (dann inklussive des ersten Elements).
	/// </summary>
	ItemHandle left;

	/// <summary>
	/// Rechte Grenze des Bereichs. Der Bereich endet mit dem Element links vom mit 'right' referenzierten Element. Das 
	/// referenzierte Element selbst gehört also nicht zum Bereich. Wenn 'right' ungültig ist, endet der Bereich mit dem 
	/// letzten Element der Liste (dann inklussive des letzten Elements).
	/// </summary>	
	ItemHandle right;

	/// <summary>
	/// Definiert, ob die in 'left' und 'right' definierten Elemente zum Bereich gehören (true) oder nicht (false).
	/// </summary>	
	bool includeBorders;
};


/// <summary>
/// Doppelt verkettete Liste beliebiger Objekte. Die Liste hat eine variable Maximalgröße.
/// Die Liste verwaltet einen internen Zeiger auf das gerade aktuelle Objekt. Dieser Zeiger kann mit Methoden wie 
/// first() oder next() durch die Liste bewegt werden.
///
/// Die Klasse ist nicht Thread-Safe! Sie ist für die Verwendung durch einen einzelnen Thread bestimmt.
///
/// Intern werden Arrays verwendet, um die Objekte zu speichern und zu verwalten. Diese Arrays haben eine bei der
/// Instantiierung definierte Größe. Sollten der Liste mehr Objekte hinzugefügt werden, als die internen Arrays
/// fassen können, so führt dies zu einer Fehlermeldung.
/// </summary>
template<typename T>
class CuDoubleLinkedList1
{
public:

	/// <summary>
	/// Erzeugt eine leere Liste, die keinerlei Speicherplatz bereit stellt und die damit unbrauchbar ist.
	/// </summary>
	CU_HSTDEV CuDoubleLinkedList1();

	/// <summary>
	/// Erzeugt eine leere Liste.
	/// </summary>
	CU_HSTDEV CuDoubleLinkedList1(int capacity);

	/// <summary>
	/// Kopiert eine Liste (deep-copy).
	/// </summary>
	CU_HSTDEV CuDoubleLinkedList1(const CuDoubleLinkedList1 &other);

	/// <summary>
	/// Destruktor.
	/// </summary>
	CU_HSTDEV ~CuDoubleLinkedList1();

	CuDoubleLinkedList1<T>& operator=(const CuDoubleLinkedList1<T> &rhs);

	bool operator==(CuDoubleLinkedList1<T> &rhs);
	bool operator!=(CuDoubleLinkedList1<T> &rhs);

	/// <summary>
	/// Erstellt beim ersten Aufruf eine Kopie des Objekts im Speicher der GPU (Device-Memory). Alle weiteren Aufrufe 
	/// liefern einen Pointer auf diese Kopie.
	/// </summary>			
	/// <returns>Pointer auf das Device-Objekt.</returns>
	CuDoubleLinkedList1<T>* getDevPtr();

	/// <summary>
	/// Überschreibt die Daten des Objekts mit denen des Objekts im Speicher der GPU (Device-Memory).
	/// </summary>			
	void copyToHost();

	/// <summary>
	/// Überschreibt die Daten des Objekts im Speicher der GPU (Device-Memory) mit denen des Objekts im Speicher des Hosts.
	/// </summary>			
	void copyToDevice();

	/// <summary>
	/// Liefert ein Handle, das dem aktuellen Element entspricht. Mit diesem Handle kann das Element später via 
	/// 'setCurrentPosition()' wieder zum aktuellen Element gemacht werden.
	/// Das Handle bleibt gültig, bis das betreffende Element aus der Liste entfernt wird. Ein ungültig gewordenes Handle 
	/// kann wieder gültig werden, wenn der entsprechende Speicherplatz für ein anderes Element wiederverwendet wird.
	/// </summary>	
	/// <returns>Handle für das aktuelle Element. Falls die Liste leer ist wird ein ungültiges Handle geliefert ('invalid').</returns>
	CU_HSTDEV ItemHandle getCurrentPosition() const;

	/// <summary>
	/// Setzt die aktuelle Position innerhalb der Liste auf das dem Handle entsprechende Element.
	/// </summary>	
	/// <returns>True, wenn das Handle gültig war und auf einen belegten Speicherplatz verweist. Andernfalls false.</returns>
	CU_HSTDEV bool setCurrentPosition(ItemHandle position);

	/// <summary>
	/// Prüft, ob die Liste leer ist.
	/// </summary>
	/// <returns>Liefert TRUE, wenn die Liste leer ist. Andernfalls FALSE.</returns>
	CU_HSTDEV bool isEmpty() const;

	/// <summary>
	/// Prüft, ob das aktuelle Element das erste Element der Liste ist.
	/// </summary>
	/// <returns>TRUE: Das aktuelle Element ist das erste Element der Liste (es steht also an erster Stelle in der Liste). Andernfalls: FALSE.</returns>
	CU_HSTDEV bool isFirst() const;

	/// <summary>
	/// Prüft, ob das aktuelle Element das letzte Element der Liste ist.
	/// </summary>
	/// <returns>TRUE: Das aktuelle Element ist das letzte Element der Liste (es steht also an letzter Stelle in der Liste). Andernfalls: FALSE.</returns>
	CU_HSTDEV bool isLast() const;

	/// <summary>
	/// Prüft, ob das aktuelle Element einen Vorgänger hat.
	/// </summary>
	/// <returns>TRUE: Das aktuelle Element hat einen Vorgänger. Es ist also nicht das erste Element in der Liste.</returns>
	CU_HSTDEV bool hasPrev() const;

	/// <summary>
	/// Prüft, ob das aktuelle Element einen Nachfolger hat.
	/// </summary>
	/// <returns>TRUE: Das aktuelle Element hat einen Nachfolger. Es ist also nicht das letzte Element in der Liste.</returns>
	CU_HSTDEV bool hasNext() const;

	/// <summary>
	/// Macht das erste Element der Liste zum aktuellen Element.
	/// </summary>
	/// <returns>True, wenn die Liste nicht leer ist. Andernfalls false.</returns>
	CU_HSTDEV bool gotoFirst();

	/// <summary>
	/// Macht das letzte Element der Liste zum aktuellen Element.
	/// </summary>
	/// <returns>True, wenn die Liste nicht leer ist. Andernfalls false.</returns>
	CU_HSTDEV bool gotoLast();

	/// <summary>
	/// Macht das nächste Element der Liste zum aktuellen Element.
	/// </summary>
	/// <returns>True, wenn es ein nächstes Element gibt. Andernfalls false.</returns>
	CU_HSTDEV bool gotoNext();

	/// <summary>
	/// Macht das vorhergehende Element der Liste zum aktuellen Element.
	/// </summary>
	/// <returns>True, wenn es ein vorhergehendes Element gibt. Andernfalls false.</returns>
	CU_HSTDEV bool gotoPrev();

	/// <summary>
	/// Liefert das erste Element der Liste und macht es gleichzeitig zum aktuellen Element.
	/// </summary>
	/// <param name="result">Referenz auf ein Objekt des Typs T, dass das Ergebnis aufnimmt. Bleibt unverändert, falls die Liste leer ist.</param>
	/// <returns>Liefert die im Parameter 'result' übergebene Referenz zurück.</returns>
	CU_HSTDEV T& first(T& result);

	/// <summary>
	/// Liefert das erste Element der Liste und macht es gleichzeitig zum aktuellen Element.
	/// </summary>
	/// <returns>Referenz auf das erste Element. Ergebnis ist undefiniert, falls die Liste leer ist.</returns>
	CU_HSTDEV T& first();

	/// <summary>
	/// Liefert das letzte Element der Liste und macht es gleichzeitig zum aktuellen Element.
	/// </summary>
	/// <param name="result">Referenz auf ein Objekt des Typs T, dass das Ergebnis aufnimmt. Bleibt unverändert, falls die Liste leer ist.</param>
	/// <returns>Liefert die im Parameter 'result' übergebene Referenz zurück.</returns>
	CU_HSTDEV T& last(T& result);

	/// <summary>
	/// Liefert das letzte Element der Liste und macht es gleichzeitig zum aktuellen Element.
	/// </summary>
	/// <returns>Referenz auf das letzte Element. Ergebnis ist undefiniert, falls die Liste leer ist.</returns>
	CU_HSTDEV T& last();

	/// <summary>
	/// Liefert das dem aktuellen Element vorhergehende Element und macht es gleichzeitig zum aktuellen Element.
	/// </summary>
	/// <param name="result">Referenz auf ein Objekt des Typs T, dass das Ergebnis aufnimmt. Bleibt unverändert, falls die Liste leer ist oder falls das aktuelle Element das erste Element der Liste ist.</param>
	/// <returns>Liefert die im Parameter 'result' übergebene Referenz zurück.</returns>
	CU_HSTDEV T& prev(T& result);

	/// <summary>
	/// Liefert das vorhergehende Element der Liste und macht es gleichzeitig zum aktuellen Element.
	/// </summary>
	/// <returns>Referenz auf das vorhergehende Element. Ergebnis ist undefiniert, falls die Liste leer ist.</returns>
	CU_HSTDEV T& prev();

	/// <summary>
	/// Liefert das dem aktuellen Element.
	/// </summary>
	/// <param name="result">Referenz auf ein Objekt des Typs T, dass das Ergebnis aufnimmt. Bleibt unverändert, falls die Liste leer ist.</param>
	/// <returns>Liefert die im Parameter 'result' übergebene Referenz zurück.</returns>
	CU_HSTDEV T& curr(T& result) const;

	/// <summary>
	/// Liefert das aktuelle Element. 
	/// </summary>
	/// <returns>Referenz auf das aktuelle Element. Ergebnis ist undefiniert, falls die Liste leer ist.</returns>
	CU_HSTDEV T& curr() const;

	/// <summary>
	/// Liefert das dem aktuellen Element folgende Element und macht es gleichzeitig zum aktuellen Element.
	/// </summary>
	/// <param name="result">Referenz auf ein Objekt des Typs T, dass das Ergebnis aufnimmt. Bleibt unverändert, falls die Liste leer ist oder falls das aktuelle Element das letzte Element der Liste ist.</param>
	/// <returns>Liefert die im Parameter 'result' übergebene Referenz zurück.</returns>
	CU_HSTDEV T& next(T& result);

	/// <summary>
	/// Liefert das nächste Element der Liste und macht es gleichzeitig zum aktuellen Element.
	/// </summary>
	/// <returns>Referenz auf das nächste Element. Ergebnis ist undefiniert, falls die Liste leer ist.</returns>
	CU_HSTDEV T& next();

	/// <summary>
	/// Liefert eine neue (Teil-)Liste zurück, indem die Listenelemente in eine neue Liste kopiert werden.
	/// </summary>
	/// <returns></returns>
	shared_ptr<CuDoubleLinkedList1<T>> copy(ListRegion region);

	/// <summary>
	/// Kopiert einen Teil der Liste und hängt diese Kopie an 'result' an.
	/// </summary>
	/// <returns>Liefert 'result' zurück.</returns>
	CU_HSTDEV CuDoubleLinkedList1<T>& copy(ListRegion region, CuDoubleLinkedList1<T> &result);

	/// <summary>
	/// Fügt vor dem aktuellen Element ein neues Element ein. 
	/// </summary>
	/// <param name="newItem">Das neue Element, das der Liste hinzugefügt werden soll.</param>
	CU_HSTDEV void insertItemBeforeCurrent(const T& newItem);

	/// <summary>
	/// Fügt nach dem aktuellen Element ein neues Element ein. 
	/// </summary>
	/// <param name="newItem">Das neue Element, das der Liste hinzugefügt werden soll.</param>
	CU_HSTDEV void insertItemAfterCurrent(const T& newItem);

	/// <summary>
	/// Fügt ein neues Element an das Ende der Liste an.
	/// </summary>
	/// <param name="newItem">Das neue Element, das der Liste hinzugefügt werden soll.</param>
	CU_HSTDEV void appendItem(const T& newItem);

	/// <summary>
	/// Entfernt das aktuelle Element aus der Liste. Das dem gelöschten Element folgende Element wird 
	/// danach zum aktuellen Element.
	/// </summary>
	CU_HSTDEV void removeCurrentItem();

	/// <summary>
	/// Fügt vor dem aktuellen Element eine Liste neuer Elemente ein. 
	/// </summary>
	/// <param name="newItems">Die Liste neuer Elemente.</param>
	CU_HSTDEV void insertListBeforeCurrent(CuDoubleLinkedList1<T> &newItems);

	/// <summary>
	/// Fügt nach dem aktuellen Element eine Liste neuer Elemente ein. 
	/// </summary>
	/// <param name="newItems">Die Liste neuer Elemente.</param>
	CU_HSTDEV void insertListAfterCurrent(CuDoubleLinkedList1<T> &newItems);

	/// <summary>
	/// Fügt eine Liste neuer Elemente an das Ende der Liste an.
	/// </summary>
	/// <param name="newItems">Die Liste neuer Elemente.</param>
	CU_HSTDEV void appendList(CuDoubleLinkedList1<T> &newItems);

	/// <summary>
	/// Entfernt mehrere Elemente aus der Liste. 
	/// Falls dabei das gerade aktuelle Element gelöscht wird, wird das den gelöschten
	/// Elementen folgende Element zum neuen aktuellen Element. Falls den gelöschten 
	/// Elementen kein Element mehr folgt, wird das ihnen vorhergehende Element zum 
	/// neuen aktuellen Element.
	/// </summary>
	/// <param name="region">Bestimmt den zu löschenden Bereichs. Default: Lösche alle Elemente der Liste.</param>
	CU_HSTDEV void removeItems(ListRegion region = ListRegion());

	/// <summary>
	/// Besucht jedes Listenelement innerhalb des durch 'region' definierten Bereichs und ruft die in 'visitor' übergebene
	/// Funktion auf. Dabei werden die Elemente in der korrekten Reihenfolge besucht (ausgehend vom Listenanfang hin zum
	/// Listenende).
	/// </summary>
	/// <param name="visitor">Funktion, die für jedes Element aufgerufen wird. Liefert 'visitor' FALSE zurück, werden keine weiteren Elemente aufgesucht.</param>	
	/// <param name="region">Bestimmt den Bereich, innerhalb dessen Elemente besucht werden. Default: Besuche alle Elemente der Liste.</param>	
	template<typename D1, typename D2>
	CU_HSTDEV void visitItems(bool (visitor)(T&, D1, D2), D1 data1, D2 data2, ListRegion region = ListRegion());

	/// <summary>
	/// Besucht jedes Listenelement innerhalb des durch 'region' definierten Bereichs und ruft die in 'visitor' übergebene
	/// Funktion auf. Dabei werden die Elemente in umgekehrter Reihenfolge besucht (ausgehend vom Listenende hin zum
	/// Listenanfang).
	/// </summary>
	/// <param name="visitor">Funktion, die für jedes Element aufgerufen wird. Liefert 'visitor' FALSE zurück, werden keine weiteren Elemente aufgesucht.</param>	
	/// <param name="region">Bestimmt den Bereich, innerhalb dessen Elemente besucht werden. Default: Besuche alle Elemente der Liste.</param>	
	template<typename D1, typename D2>
	CU_HSTDEV void visitItemsReversedOrder(bool (visitor)(T&, D1, D2), D1 data1, D2 data2, ListRegion region = ListRegion());

	/// <summary>
	/// Verwendet den Gleichheitsoperator (==), um ein Objekt in der Liste zu finden und es zum aktuellen 
	/// Element zu machen.
	/// </summary>
	/// <param name="item">Das Objekt, das gesucht werden soll.</param>
	/// <returns>TRUE, wenn das Objekt in der Liste gefunden wurde und nun das aktuelle Element ist. Sonst FALSE.</returns>
	CU_HSTDEV bool find(const T& item);

	/// <summary>
	/// Liefert die aktuelle Anzahl der in der Liste enhaltenen Elemente. 
	/// </summary>
	/// <returns>Anzahl der Listenelemente.</returns>
	CU_HSTDEV int getSize() const;

	/// <summary>
	/// Liefert die Anzahl der Elemente, welche die Liste maximal aufnehmen kann.
	/// </summary>
	/// <returns>Maximale Anzahl der Listenelemente.</returns>
	CU_HSTDEV int getCapacity() const;

private:
	CU_HSTDEV void addNewItem(int prevIdx, int nextIdx, const T& newItem);
	CU_HSTDEV void deleteItem(int idx);

	T *_data;			// Payload des Elements.
	short *_prev;		// Index des vorhergehenden Elements.
	short *_next;		// Index des nächsten Elements.
	bool *_occupied;	// Flag: TRUE, falls Speicherstelle bereits belegt ist.
	short _root;		// Index des Wurzelelements.
	short _current;		// Index des aktuellen Elements.
	short _size;		// Aktuelle Anzahl der Elemente in der Liste.
	short _capacity;	// Maximale Anzahl der Listenelemente.

	CuDoubleLinkedList1<T>* _devPtr;
};


template<typename T>
CU_HSTDEV CuDoubleLinkedList1<T>::CuDoubleLinkedList1()
	: _root(-1), _current(-1), _size(0), _capacity(0), _data(0), _prev(0), _next(0), _occupied(0), _devPtr(0)
{
}


template<typename T>
CU_HSTDEV CuDoubleLinkedList1<T>::CuDoubleLinkedList1(int capacity)
	: _root(-1), _current(-1), _size(0), _capacity(capacity), _data(0), _prev(0), _next(0), _occupied(0), _devPtr(0)
{
	if (capacity <= 0 || capacity >= SHRT_MAX) {
#ifdef __CUDACC__
		printf("CuDoubleLinkedList1<T>::CuDoubleLinkedList1(): Parameter capacity ist unzulässig!\n");
		return;
#else
		throw std::invalid_argument("CuDoubleLinkedList1<T>::CuDoubleLinkedList1(): Parameter capacity ist unzulässig!");
#endif
	}

	_data = new T[capacity];
	_prev = new short[capacity];
	_next = new short[capacity];
	_occupied = new bool[capacity];
	memset(_occupied, 0, capacity); // Alles auf FALSE setzen.
}


template<typename T>
CU_HSTDEV CuDoubleLinkedList1<T>::CuDoubleLinkedList1(const CuDoubleLinkedList1 &other)
	: _root(other._root), _current(other._current), _size(other._size), _capacity(other._capacity),
	_data(0), _prev(0), _next(0), _occupied(0), _devPtr(0)
{
	_data = new T[_capacity];
	_prev = new short[_capacity];
	_next = new short[_capacity];
	_occupied = new bool[_capacity];
	memcpy(_data, other._data, sizeof(T) * _capacity);
	memcpy(_prev, other._prev, sizeof(short) * _capacity);
	memcpy(_next, other._next, sizeof(short) * _capacity);
	memcpy(_occupied, other._occupied, sizeof(bool) * _capacity);
}


template<typename T>
CU_HSTDEV CuDoubleLinkedList1<T>::~CuDoubleLinkedList1()
{
	if (_data) delete[] _data;
	if (_prev) delete[] _prev;
	if (_next) delete[] _next;
	if (_occupied) delete[] _occupied;
#ifdef __CUDACC__
	if (_devPtr) {
		CuDoubleLinkedList1<T> *tempDevPtr = _devPtr;
		CUDA_CHECK(cudaMemcpy(this, _devPtr, sizeof(CuVector1<T>), cudaMemcpyDeviceToHost));
		if (_data) {
			CUDA_CHECK(wbCudaFree(__FILE__, __LINE__, _data));
			CUDA_CHECK(wbCudaFree(__FILE__, __LINE__, _prev));
			CUDA_CHECK(wbCudaFree(__FILE__, __LINE__, _next));
			CUDA_CHECK(wbCudaFree(__FILE__, __LINE__, _occupied));
		}
		CUDA_CHECK(wbCudaFree(__FILE__, __LINE__, tempDevPtr));
	}
#endif
}


template<typename T>
CuDoubleLinkedList1<T>& CuDoubleLinkedList1<T>::operator=(const CuDoubleLinkedList1<T> &rhs)
{
	if (this != &rhs) {
		if (_data) delete[] _data;
		if (_prev) delete[] _prev;
		if (_next) delete[] _next;
		if (_occupied) delete[] _occupied;
#ifdef __CUDACC__
		if (_devPtr) {
			CuDoubleLinkedList1<T> *tempDevPtr = _devPtr;
			CUDA_CHECK(cudaMemcpy(this, _devPtr, sizeof(CuVector1<T>), cudaMemcpyDeviceToHost));
			if (_data) {
				CUDA_CHECK(wbCudaFree(__FILE__, __LINE__, _data));
				CUDA_CHECK(wbCudaFree(__FILE__, __LINE__, _prev));
				CUDA_CHECK(wbCudaFree(__FILE__, __LINE__, _next));
				CUDA_CHECK(wbCudaFree(__FILE__, __LINE__, _occupied));
			}
			CUDA_CHECK(wbCudaFree(__FILE__, __LINE__, tempDevPtr));
		}
#endif

		_root = rhs._root;
		_current = rhs._current;
		_size = rhs._size;
		_capacity = rhs._capacity;

		_data = new T[_capacity];
		_prev = new short[_capacity];
		_next = new short[_capacity];
		_occupied = new bool[_capacity];

		memcpy(_data, rhs._data, sizeof(T) * _capacity);
		memcpy(_prev, rhs._prev, sizeof(short) * _capacity);
		memcpy(_next, rhs._next, sizeof(short) * _capacity);
		memcpy(_occupied, rhs._occupied, sizeof(bool) * _capacity);
	}

	return *this;
}


template<typename T>
bool CuDoubleLinkedList1<T>::operator==(CuDoubleLinkedList1<T> &rhs)
{
	if (this == &rhs) return true;

	if (getSize() != rhs.getSize()) return false;
	if (isEmpty()) return true;

	ItemHandle pos = getCurrentPosition();
	ItemHandle posRhs = rhs.getCurrentPosition();

	gotoFirst();
	rhs.gotoFirst();

	do {
		if (curr() != rhs.curr()) {
			setCurrentPosition(pos);
			rhs.setCurrentPosition(posRhs);
			return false;
		}
		gotoNext();
	} while (rhs.gotoNext());

	setCurrentPosition(pos);
	rhs.setCurrentPosition(posRhs);

	return true;
}


template<typename T>
bool CuDoubleLinkedList1<T>::operator!=(CuDoubleLinkedList1<T> &rhs)
{
	return !(*this == rhs);
}


template<typename T>
CuDoubleLinkedList1<T>* CuDoubleLinkedList1<T>::getDevPtr()
{
	assert(false);
	return 0;
}


template<typename T>
void CuDoubleLinkedList1<T>::copyToHost()
{
	assert(false);
}


template<typename T>
void CuDoubleLinkedList1<T>::copyToDevice()
{
	assert(false);
}


template<typename T>
CU_HSTDEV ItemHandle CuDoubleLinkedList1<T>::getCurrentPosition() const
{
	assert(_current >= 0);
	if (_current >= 0) {
		return ItemHandle(_current);
	}
	return ItemHandle::invalid();
}


template<typename T>
CU_HSTDEV bool CuDoubleLinkedList1<T>::setCurrentPosition(ItemHandle position)
{
	assert(position.isValid());
	assert(_occupied[(short)position]);
	if (position.isValid() && _occupied[(short)position]) {
		_current = (short)position;
		return true;
	}
	return false;
}


template<typename T>
CU_HSTDEV bool CuDoubleLinkedList1<T>::isEmpty() const
{
	return _size == 0;
}


template<typename T>
CU_HSTDEV bool CuDoubleLinkedList1<T>::isFirst() const
{
	if (!isEmpty()) {
		assert(_current >= 0);
		assert(_root >= 0);
		assert(_size > 0);
		return _current == _root;
	}
	return false;
}


template<typename T>
CU_HSTDEV bool CuDoubleLinkedList1<T>::isLast() const
{
	if (!isEmpty()) {
		assert(_current >= 0);
		assert(_root >= 0);
		assert(_size > 0);
		return _next[_current] < 0;
	}
	return false;
}


template<typename T>
CU_HSTDEV bool CuDoubleLinkedList1<T>::hasPrev() const
{
	if (!isEmpty()) {
		assert(_current >= 0);
		assert(_root >= 0);
		assert(_size > 0);
		return _prev[_current] >= 0;
	}
	return false;
}


template<typename T>
CU_HSTDEV bool CuDoubleLinkedList1<T>::hasNext() const
{
	if (!isEmpty()) {
		assert(_current >= 0);
		assert(_root >= 0);
		assert(_size > 0);
		return _next[_current] >= 0;
	}
	return false;
}


template<typename T>
CU_HSTDEV bool CuDoubleLinkedList1<T>::gotoFirst()
{
	if (!isEmpty()) {
		_current = _root;
		assert(_current >= 0);
		assert(_root >= 0);
		assert(_size > 0);
		return true;
	}
	return false;
}


template<typename T>
CU_HSTDEV bool CuDoubleLinkedList1<T>::gotoLast()
{
	if (!isEmpty()) {
		while (gotoNext());
		return true;
	}
	return false;
}


template<typename T>
CU_HSTDEV bool CuDoubleLinkedList1<T>::gotoNext()
{
	if (hasNext()) {
		assert(_current >= 0);
		assert(_root >= 0);
		assert(_size > 0);

		_current = _next[_current];
		assert(_current >= 0);

		return true;
	}
	return false;
}


template<typename T>
CU_HSTDEV bool CuDoubleLinkedList1<T>::gotoPrev()
{
	if (hasPrev()) {
		assert(_current >= 0);

		_current = _prev[_current];
		assert(_current >= 0);

		return true;
	}
	return false;
}


template<typename T>
CU_HSTDEV T& CuDoubleLinkedList1<T>::last(T& result)
{
	if (!isEmpty()) {
		gotoLast();

		assert(_current >= 0);

		result = _data[_current];
	}
	return result;
}


template<typename T>
CU_HSTDEV T& CuDoubleLinkedList1<T>::prev(T& result)
{
	if (hasPrev()) {

		assert(_current >= 0);

		_current = _prev[_current];

		assert(_current >= 0);

		result = _data[_current];
	}
	return result;
}


template<typename T>
CU_HSTDEV T& CuDoubleLinkedList1<T>::curr(T& result) const
{
	if (!isEmpty()) {

		assert(_current >= 0);

		result = _data[_current];
	}
	return result;
}


template<typename T>
CU_HSTDEV T& CuDoubleLinkedList1<T>::next(T& result)
{
	if (hasNext()) {
		assert(_current >= 0);
		_current = _next[_current];
		assert(_current >= 0);
		result = _data[_current];
	}
	return result;
}




template<typename T>
CU_HSTDEV T& CuDoubleLinkedList1<T>::first(T& result)
{
	if (!isEmpty()) {
		assert(_root >= 0);

		_current = _root;
		result = _data[_root];
	}
	return result;
}


template<typename T>
CU_HSTDEV T& CuDoubleLinkedList1<T>::first()
{
	assert(_root >= 0);
	_current = _root;
	return _data[_root];
}


template<typename T>
CU_HSTDEV T& CuDoubleLinkedList1<T>::last()
{
	gotoLast();
	assert(_current >= 0);
	return _data[_current];
}


template<typename T>
CU_HSTDEV T& CuDoubleLinkedList1<T>::prev()
{
	assert(_current >= 0);
	_current = _prev[_current];
	assert(_current >= 0);
	return _data[_current];
}


template<typename T>
CU_HSTDEV T& CuDoubleLinkedList1<T>::curr() const
{
	assert(_current >= 0);
	return _data[_current];
}


template<typename T>
CU_HSTDEV T& CuDoubleLinkedList1<T>::next()
{
	assert(_current >= 0);
	_current = _next[_current];
	assert(_current >= 0);
	return _data[_current];
}


template<typename T>
shared_ptr<CuDoubleLinkedList1<T>> CuDoubleLinkedList1<T>::copy(ListRegion region)
{
	assert(!region.left.isValid() || _occupied[(short)region.left]);
	if ((region.left.isValid() && !_occupied[(short)region.left])) {
		return 0;
	}

	assert(!region.right.isValid() || _occupied[(short)region.right]);
	if ((region.right.isValid() && !_occupied[(short)region.right])) {
		return 0;
	}

	CuDoubleLinkedList1<T> *retValPtr = new CuDoubleLinkedList1<T>(getSize());
	shared_ptr<CuDoubleLinkedList1<T>> retVal(retValPtr);

	copy(region, *retValPtr);

	//if (!region.includeBorders && region.left.isValid() && region.left == region.right) {
	//	// In diesem Fall ist nichts zu kopieren. Gib eine leere Liste zurück.
	//	return retVal;
	//}

	//if (!isEmpty()) {
	//	if (region.left.isValid()) {
	//		setCurrentPosition(region.left);
	//		if (!region.includeBorders) {
	//			if (!gotoNext()) return retVal;
	//		}
	//	}
	//	else {
	//		gotoFirst();
	//	}

	//	if (!region.includeBorders && region.right.isValid() && region.right == getCurrentPosition()) {
	//		// In diesem Fall ist nichts zu kopieren. Gib eine leere Liste zurück.
	//		return retVal;
	//	}

	//	do {
	//		retVal->appendItem(curr());
	//	} while (gotoNext() && getCurrentPosition() != region.right);

	//	if (region.includeBorders && getCurrentPosition() == region.right) {
	//		retVal->appendItem(curr());
	//	}
	//}

	return retVal;
}


template<typename T>
CU_HSTDEV CuDoubleLinkedList1<T>& CuDoubleLinkedList1<T>::copy(ListRegion region, CuDoubleLinkedList1<T> &result)
{
	assert(!region.left.isValid() || _occupied[(short)region.left]);
	if ((region.left.isValid() && !_occupied[(short)region.left])) {
		return result;
	}

	assert(!region.right.isValid() || _occupied[(short)region.right]);
	if ((region.right.isValid() && !_occupied[(short)region.right])) {
		return result;
	}

	if (!region.includeBorders && region.left.isValid() && region.left == region.right) {
		// In diesem Fall ist nichts zu kopieren. Gib die unveränderte Liste zurück.
		return result;
	}

	if (!isEmpty()) {
		if (region.left.isValid()) {
			setCurrentPosition(region.left);
			if (!region.includeBorders) {
				if (!gotoNext()) return result;
			}
		}
		else {
			gotoFirst();
		}

		if (!region.includeBorders && region.right.isValid() && region.right == getCurrentPosition()) {
			// In diesem Fall ist nichts zu kopieren. Gib die unveränderte Liste zurück.
			return result;
		}

		do {
			result.appendItem(curr());
		} while (gotoNext() && getCurrentPosition() != region.right);

		if (region.includeBorders && getCurrentPosition() == region.right) {
			result.appendItem(curr());
		}
	}

	return result;
}


template<typename T>
CU_HSTDEV void CuDoubleLinkedList1<T>::insertItemBeforeCurrent(const T& newItem)
{
	if (hasPrev()) {
		addNewItem(_prev[_current], _current, newItem);
	}
	else {
		// Entweder ist die Liste noch leer oder das aktuelle Element steht in der Liste an erster Stelle.
		addNewItem(-1, _current, newItem);
	}
}


template<typename T>
CU_HSTDEV void CuDoubleLinkedList1<T>::insertItemAfterCurrent(const T& newItem)
{
	if (hasNext()) {
		addNewItem(_current, _next[_current], newItem);
	}
	else {
		// Entweder ist die Liste noch leer oder das aktuelle Element steht in der Liste an letzter Stelle.
		addNewItem(_current, -1, newItem);
	}
}


template<typename T>
CU_HSTDEV void CuDoubleLinkedList1<T>::appendItem(const T& newItem)
{
	if (isEmpty()) {
		addNewItem(-1, -1, newItem);
	}
	else {
		short saveCurrent = _current;
		gotoLast();
		insertItemAfterCurrent(newItem);
		_current = saveCurrent;
	}
}


template<typename T>
CU_HSTDEV void CuDoubleLinkedList1<T>::removeCurrentItem()
{
	if (isEmpty()) return;

	short nextIdx = _next[_current];
	short prevIdx = _prev[_current];

	if (isFirst()) {
		if (hasNext()) {
			_prev[nextIdx] = -1;
			deleteItem(_current);
		}
		else {
			_root = -1;
			_current = -1;
			_size = 0;
		}
	}
	else if (isLast()) {
		_next[prevIdx] = -1;
		deleteItem(_current);
		_current = prevIdx;
	}
	else {
		_next[prevIdx] = nextIdx;
		_prev[nextIdx] = prevIdx;
		deleteItem(_current);
		_current = nextIdx;
	}
}


template<typename T>
CU_HSTDEV bool CuDoubleLinkedList1<T>::find(const T& item)
{
	if (!isEmpty()) {
		int saveCurrent = _current;
		T currItem = first(currItem);
		if (item == currItem) return true;
		while (hasNext()) {
			currItem = next(currItem);
			if (item == currItem) return true;
		}
		_current = saveCurrent;
	}
	return false;
}


template<typename T>
CU_HSTDEV void CuDoubleLinkedList1<T>::addNewItem(int prevIdx, int nextIdx, const T& newItem)
{
	if ((int)_size >= _capacity)
	{
		assert(false);
#ifdef __CUDACC__
		printf("CuDoubleLinkedList1<T>::addNewItem(): Liste ist voll!\n");
		return;
#else
		throw std::runtime_error("CuDoubleLinkedList1<T>::addNewItem(): Liste ist voll!");
#endif
	}

	// Finde eine freie Speicherstelle.
	short idx = _size;
	while (_occupied[idx] && idx < _capacity) idx++; // Im Bereich >=_size ist die Wahrscheinlichkeit gross, eine freie Stelle zu finden.
	if (idx == _capacity) {
		// Noch keine freie Stelle gefunden. Jetzt die durch Löschen freigewordenen Stellen wiederverwenden.
		idx = 0;
		while (_occupied[idx] && idx < _capacity) idx++;
		if (idx == _capacity) { // Offenbar liegt ein Fehler vor: _size oder _occupied sind anscheinend nicht korrekt gepflegt.
#ifdef __CUDACC__
			printf("CuDoubleLinkedList1<T>::addNewItem(): Interner Programmfehler!\n");
			return;
#else
			throw std::runtime_error("CuDoubleLinkedList1<T>::addNewItem(): Interner Programmfehler!");
#endif
	}
}

	_size++;
	_occupied[idx] = true;

	_data[idx] = newItem;
	_prev[idx] = prevIdx;
	_next[idx] = nextIdx;

	if (nextIdx >= 0) {
		_prev[nextIdx] = idx;
	}

	if (prevIdx >= 0) {
		_next[prevIdx] = idx;
	}
	else {
		if (_root < 0) {
			_current = idx;
		}
		_root = idx;
	}

}


template<typename T>
CU_HSTDEV void CuDoubleLinkedList1<T>::deleteItem(int idx)
{
	assert(idx >= 0);
	assert(idx < _capacity);
	assert(_occupied[idx]);

	if (idx < 0 || idx >= _capacity) {
#ifdef __CUDACC__
		printf("CuDoubleLinkedList1<T>::deleteItem(): CuDoubleLinkedList1<T>::deleteItem(): Index ist unzulässig!\n");
		return;
#else
		throw std::invalid_argument("CuDoubleLinkedList1<T>::deleteItem(): Index ist unzulässig!");
#endif
}

	if (!_occupied[idx]) {
#ifdef __CUDACC__
		printf("CuDoubleLinkedList1<T>::deleteItem(): CuDoubleLinkedList1<T>::deleteItem(): Das Element existiert nicht!\n");
		return;
#else
		throw std::invalid_argument("CuDoubleLinkedList1<T>::deleteItem(): Das Element existiert nicht!");
#endif
	}

	short nextItem = _next[idx];
	short prevItem = _prev[idx];

	if (nextItem >= 0) {
		if (prevItem >= 0) {
			// Das zu löschende Element ist ein Element mitten in der Liste.
			_prev[nextItem] = prevItem;
			_next[prevItem] = nextItem;
			if (_current == idx) {
				_current = nextItem;
			}
		}
		else {
			// Das zu löschende Element ist das erste Element der Liste.
			_prev[nextItem] = -1;
			_root = nextItem;
			if (_current == idx) {
				_current = _root;
			}
		}
	}
	else {
		if (prevItem >= 0) {
			// Das zu löschende Element ist das letzte Element der Liste (also das Element, das am Ende der Liste steht).
			_next[prevItem] = -1;
			if (_current == idx) {
				_current = prevItem;
			}
		}
		else {
			// Das zu löschende Element ist das einzige Element der Liste.
			assert(_size == 1);
			_root = -1;
			_current = -1;
		}
	}

	_size--;
	_occupied[idx] = false;
}


template<typename T>
CU_HSTDEV int CuDoubleLinkedList1<T>::getSize() const
{
	return _size;
}


template<typename T>
CU_HSTDEV int CuDoubleLinkedList1<T>::getCapacity() const
{
	return _capacity;
}


template<typename T>
CU_HSTDEV void CuDoubleLinkedList1<T>::insertListBeforeCurrent(CuDoubleLinkedList1<T> &newItems)
{
	assert(!newItems.isEmpty());
	ItemHandle oldPos = newItems.getCurrentPosition();
	if (newItems.gotoFirst()) {
		do {
			insertItemBeforeCurrent(newItems.curr());
		} while (newItems.gotoNext());
	}
	newItems.setCurrentPosition(oldPos);
}


template<typename T>
CU_HSTDEV void CuDoubleLinkedList1<T>::insertListAfterCurrent(CuDoubleLinkedList1<T> &newItems)
{
	assert(!newItems.isEmpty());
	ItemHandle oldPos = newItems.getCurrentPosition();
	if (newItems.gotoLast()) {
		// Das aktuelle Element der Liste ändert sich durch das Einfügen nicht.
		// Daher müssen die neuen Elemente in umgekehrter Reihenfolge eingefügt werden.
		do {
			insertItemAfterCurrent(newItems.curr());
		} while (newItems.gotoPrev());
	}
	newItems.setCurrentPosition(oldPos);
}


template<typename T>
CU_HSTDEV void CuDoubleLinkedList1<T>::appendList(CuDoubleLinkedList1<T> &newItems)
{
	assert(!newItems.isEmpty());
	if (newItems.isEmpty()) return;

	if (isEmpty()) {
		newItems.gotoFirst();
		do {
			appendItem(newItems.curr());
		} while (newItems.gotoNext());
	}
	else {
		ItemHandle oldPos = getCurrentPosition();
		gotoLast();
		insertListAfterCurrent(newItems);
		setCurrentPosition(oldPos);
	}
}


template<typename T>
CU_HSTDEV void CuDoubleLinkedList1<T>::removeItems(ListRegion region)
{
	if (isEmpty()) return;

	ItemHandle oldCurrent = getCurrentPosition();

	//if (region.left.isValid() && region.left == region.right) {
	//	if (region.includeBorders) {
	//		if (!setCurrentPosition(region.left)) {
	//			assert(false);
	//			return;
	//		}
	//		removeCurrentItem();
	//	}
	//	else {
	//		assert(false);
	//		return;
	//	}
	//}
	//else {

	if (region.left.isValid()) {
		if (!setCurrentPosition(region.left)) {
			assert(false);
			return;
		}
		if (!region.includeBorders) {
			if (!gotoNext()) {
				assert(false);
				return;
			}
		}
	}
	else {
		gotoFirst();
	}

	while (!isLast() && region.right != getCurrentPosition()) {
		removeCurrentItem();
	}

	if (!region.right.isValid() || region.includeBorders && region.right == getCurrentPosition()) {
		removeCurrentItem();
	}
	//}

	if (_occupied[(short)oldCurrent]) setCurrentPosition(oldCurrent);
}


template<typename T>
template<typename D1, typename D2>
CU_HSTDEV void CuDoubleLinkedList1<T>::visitItems(bool (visitor)(T&, D1, D2), D1 data1, D2 data2, ListRegion region)
{
	ItemHandle oldCurrent = getCurrentPosition();

	if (region.left.isValid()) {
		if (!setCurrentPosition(region.left)) {
			assert(false);
			return;
		}
		if (!region.includeBorders) {
			if (!gotoNext()) {
				// 'region.left' zeigt offenbar auf das letzte Listenelement. 
				// Es gibt keine weiteren Elemente, die besucht werden könnten.
				return;
			}
		}
	}
	else {
		gotoFirst();
	}

	bool run = true;
	while (!isLast() && region.right != getCurrentPosition()) {
		run = visitor(curr(), data1, data2);
		if (!run) break;
		if (!gotoNext()) {
			assert(false);
			return;
		}
	}

	if (run && (!region.right.isValid() || region.includeBorders && region.right == getCurrentPosition())) {
		visitor(curr(), data1, data2);
	}

	setCurrentPosition(oldCurrent);
}


template<typename T>
template<typename D1, typename D2>
CU_HSTDEV void CuDoubleLinkedList1<T>::visitItemsReversedOrder(bool (visitor)(T&, D1, D2), D1 data1, D2 data2, ListRegion region)
{
	ItemHandle oldCurrent = getCurrentPosition();

	assert(!isEmpty());
	if (isEmpty()) return;

	if (region.right.isValid()) {
		if (!setCurrentPosition(region.right)) {
			assert(false);
			return;
		}
		if (!region.includeBorders) {
			if (!gotoPrev()) {
				assert(false);
				return;
			}
		}
	}
	else {
		gotoLast();
	}

	bool run = true;
	while (!isFirst() && region.left != getCurrentPosition()) {
		run = visitor(curr(), data1, data2);
		if (!run) break;
		if (!gotoPrev()) {
			assert(false);
			return;
		}
	}

	if (run && (!region.left.isValid() || region.includeBorders && region.left == getCurrentPosition())) {
		visitor(curr(), data1, data2);
	}

	setCurrentPosition(oldCurrent);
}