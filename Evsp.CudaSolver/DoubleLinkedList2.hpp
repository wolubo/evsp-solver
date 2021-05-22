#pragma once

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda.h> 
#else
#include <stdexcept>
#endif

#include <assert.h>
#include "EVSP.BaseClasses/DeviceLaunchParameters.h"
#include "EVSP.BaseClasses/Typedefs.h"
#include "CudaCheck.h"

/// <summary>
/// Doppelt verkettete Liste beliebiger Objekte. Die Liste hat eine fixe Maximalgröße.
/// Die Liste verwaltet einen internen Zeiger auf das gerade aktuelle Objekt. Dieser Zeiger kann mit Methoden wie 
/// first() oder next() durch die Liste bewegt werden.
///
/// Die Klasse ist nicht Thread-Safe! Sie ist für die Verwendung durch einen einzelnen Thread bestimmt.
///
/// Intern werden Arrays verwendet, um die Objekte zu speichern und zu verwalten. Diese Arrays haben eine bei der
/// Instantiierung definierte Größe. Sollten der Liste mehr Objekte hinzugefügt werden, als die internen Arrays
/// fassen können, so führt dies zu einer Fehlermeldung.
/// </summary>
template<typename T, int SIZE>
class DoubleLinkedList2
{
public:

	/// <summary>
	/// Kopiert eine Liste aus dem Device- in den Host-Speicher.
	/// Falls die Liste Pointer enthält, so zeigen diese nach dem Kopieren immer noch in den Device-Speicher. Die einzelnen Objekte
	/// müssen dann in den Host-Speicher kopiert und in einer neuen Liste abgelegt werden.
	/// </summary>
	/// <returns>Zeiger auf die Kopie der Liste im Host-Speicher</returns>
	static DoubleLinkedList2<T, SIZE>* cpy2hst(const DoubleLinkedList2<T, SIZE> *devicePtr);

	/// <summary>
	/// Kopiert eine Liste aus dem Device- in den Host-Speicher.
	/// Für jedes Element wird dabei die Funktion 'translate()' aufgerufen. Auf diese Weise können bspw. Listen kopiert werden, 
	/// welche Pointer enthalten.
	/// </summary>
	/// <returns>Zeiger auf die Kopie der Liste im Host-Speicher</returns>
	static DoubleLinkedList2<T, SIZE>* cpy2hst(const DoubleLinkedList2<T, SIZE> *devicePtr, T translate(T));

	/// <summary>
	/// Lösche eine Liste aus dem Device-Speicher. Sofern die List Pointer enthält müssen die entsprechenden Objekte vorab 
	/// gelöscht werden (Liste auf den Host kopieren und dann wbCudaFree für jedes Objekt in der Liste).
	/// </summary>
	static void delOnDev(DoubleLinkedList2<T, SIZE> *devicePtr);

	/// <summary>
	/// Erzeugt eine leere Liste.
	/// </summary>
	CU_HSTDEV DoubleLinkedList2();

	CU_HSTDEV DoubleLinkedList2(const DoubleLinkedList2 &other) { assign(false); }

	/// <summary>
	/// Destruktor.
	/// </summary>
	CU_HSTDEV ~DoubleLinkedList2();

	DoubleLinkedList2<T, SIZE>& operator=(const DoubleLinkedList2<T, SIZE> &rhs) { assert(false); return *this; }

	/// <summary>
	/// Liefert einen Zeiger auf eine Kopie des Objekts im Device-Speicher. Jeder Aufruf generiert eine neue Kopie.
	/// Falls die Liste Pointer enthält müssen die entsprechenden Objekte vorab einzeln auf das Device kopiert und in 
	/// einer neuen Liste abgelegt werden.
	/// </summary>
	/// <returns>Zeiger auf die Kopie der Liste im Device-Speicher</returns>
	CU_HSTDEV DoubleLinkedList2<T, SIZE>* cpy2dev();

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
	/// Liefert das letzte Element der Liste und macht es gleichzeitig zum aktuellen Element.
	/// </summary>
	/// <param name="result">Referenz auf ein Objekt des Typs T, dass das Ergebnis aufnimmt. Bleibt unverändert, falls die Liste leer ist.</param>
	/// <returns>Liefert die im Parameter 'result' übergebene Referenz zurück.</returns>
	CU_HSTDEV T& last(T& result);

	/// <summary>
	/// Liefert das dem aktuellen Element vorhergehende Element und macht es gleichzeitig zum aktuellen Element.
	/// </summary>
	/// <param name="result">Referenz auf ein Objekt des Typs T, dass das Ergebnis aufnimmt. Bleibt unverändert, falls die Liste leer ist oder falls das aktuelle Element das erste Element der Liste ist.</param>
	/// <returns>Liefert die im Parameter 'result' übergebene Referenz zurück.</returns>
	CU_HSTDEV T& prev(T& result);

	/// <summary>
	/// Liefert das dem aktuellen Element.
	/// </summary>
	/// <param name="result">Referenz auf ein Objekt des Typs T, dass das Ergebnis aufnimmt. Bleibt unverändert, falls die Liste leer ist.</param>
	/// <returns>Liefert die im Parameter 'result' übergebene Referenz zurück.</returns>
	CU_HSTDEV T& curr(T& result) const;

	/// <summary>
	/// Liefert das dem aktuellen Element folgende Element und macht es gleichzeitig zum aktuellen Element.
	/// </summary>
	/// <param name="result">Referenz auf ein Objekt des Typs T, dass das Ergebnis aufnimmt. Bleibt unverändert, falls die Liste leer ist oder falls das aktuelle Element das letzte Element der Liste ist.</param>
	/// <returns>Liefert die im Parameter 'result' übergebene Referenz zurück.</returns>
	CU_HSTDEV T& next(T& result);

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
	CU_HSTDEV void append(const T& newItem);

	/// <summary>
	/// Entfernt das aktuelle Element aus der Liste. Das dem gelöschten Element folgende Element wird 
	/// danach zum aktuellen Element.
	/// </summary>
	CU_HSTDEV void removeCurrent();

	/// <summary>
	/// Verwendet den Gleichheitsoperator (==), um ein Objekt in der Liste zu finden und es zum aktuellen 
	/// Element zu machen.
	/// </summary>
	/// <param name="item">Das Objekt, das gesucht werden soll.</param>
	/// <returns>TRUE, wenn das Objekt in der Liste gefunden wurde und nun das aktuelle Element ist. Sonst FALSE.</returns>
	CU_HSTDEV bool find(const T& item);

	/// <summary>
	/// Liefert die Anzahl der in der Liste enhaltenen Elemente. 
	/// </summary>
	/// <returns>Anzahl der Listenelemente.</returns>
	CU_HSTDEV __int16 getSize() const;

private:
	CU_HSTDEV void addNewItem(int prevIdx, int nextIdx, const T& newItem);
	CU_HSTDEV void deleteItem(__int16 idx);

	T _data[SIZE];
	__int16 _prev[SIZE];
	__int16 _next[SIZE];
	__int16 _root;
	__int16 _current;
	__int16 _size;			// Aktuelle Anzahl der Elemente in der Liste.
};


#ifdef __CUDACC__
template<typename T, int SIZE>
DoubleLinkedList2<T, SIZE>* DoubleLinkedList2<T, SIZE>::cpy2hst(const DoubleLinkedList2<T, SIZE> *devicePtr)
{
	DoubleLinkedList2<T, SIZE>* retVal = 0;
	if (devicePtr) {
		retVal = new DoubleLinkedList2<T, SIZE>();
		CUDA_CHECK(cudaMemcpy(retVal, devicePtr, sizeof(DoubleLinkedList2<T, SIZE>), cudaMemcpyDeviceToHost));

		__int16 *prev = new __int16[retVal->_capacity];
		CUDA_CHECK(cudaMemcpy(prev, retVal->_prev, retVal->_size * sizeof(__int16), cudaMemcpyDeviceToHost));
		retVal->_prev = prev;

		__int16 *next = new __int16[retVal->_capacity];
		CUDA_CHECK(cudaMemcpy(next, retVal->_next, retVal->_size * sizeof(__int16), cudaMemcpyDeviceToHost));
		retVal->_next = next;

		T* data = new T[retVal->_capacity];
		CUDA_CHECK(cudaMemcpy(data, retVal->_data, retVal->_size * sizeof(T), cudaMemcpyDeviceToHost));
		retVal->_data = data;
	}
	return retVal;
}
#endif


#ifdef __CUDACC__
template<typename T, int SIZE>
DoubleLinkedList2<T, SIZE>* DoubleLinkedList2<T, SIZE>::cpy2hst(const DoubleLinkedList2<T, SIZE> *devicePtr, T translate(T))
{
	DoubleLinkedList2<T, SIZE>* list = cpy2hst(devicePtr);
	DoubleLinkedList2<T, SIZE>* retVal = new DoubleLinkedList2<T, SIZE>(list->_capacity, list->_growRate);
	T currendItem;
	if (!list->isEmpty()) {
		list->gotoFirst();
		do {
			currendItem = list->curr(currendItem);
			T translatedItem = translate(currendItem);
			retVal->append(translatedItem);
		} while (list->gotoNext());
	}
	delete list;
	return retVal;
}
#endif


#ifdef __CUDACC__
template<typename T, int SIZE>
void DoubleLinkedList2<T, SIZE>::delOnDev(DoubleLinkedList2<T, SIZE> *devicePtr)
{
	if (devicePtr) {
		DoubleLinkedList2<T, SIZE> temp;
		CUDA_CHECK(cudaMemcpy(&temp, devicePtr, sizeof(DoubleLinkedList2<T, SIZE>), cudaMemcpyDeviceToHost));

		if (temp._prev) CUDA_CHECK(wbCudaFree(__FILE__, __LINE__, temp._prev));
		if (temp._next) CUDA_CHECK(wbCudaFree(__FILE__, __LINE__, temp._next));
		if (temp._data) CUDA_CHECK(wbCudaFree(__FILE__, __LINE__, temp._data));
		CUDA_CHECK(wbCudaFree(__FILE__, __LINE__, devicePtr));
	}
}
#endif


template<typename T, int SIZE>
CU_HSTDEV DoubleLinkedList2<T, SIZE>::DoubleLinkedList2()
	: _root(-1), _current(-1), _size(0), _data(), _prev(), _next()
{
}


template<typename T, int SIZE>
CU_HSTDEV DoubleLinkedList2<T, SIZE>::~DoubleLinkedList2()
{
}


#ifdef __CUDACC__
template<typename T, int SIZE>
CU_HSTDEV DoubleLinkedList2<T, SIZE>* DoubleLinkedList2<T, SIZE>::cpy2dev()
{
	DoubleLinkedList2<T, SIZE> *retVal;
	CUDA_CHECK(wbCudaMalloc(__FILE__, __LINE__, (void**)&retVal, sizeof(DoubleLinkedList2<T, SIZE>)));
	
	CUDA_CHECK(cudaMemcpy(retVal, this, sizeof(DoubleLinkedList2<T, SIZE>), cudaMemcpyHostToDevice));
	return retVal;
}
#endif


template<typename T, int SIZE>
CU_HSTDEV bool DoubleLinkedList2<T, SIZE>::isEmpty() const
{
	return _root < 0;
}


template<typename T, int SIZE>
CU_HSTDEV bool DoubleLinkedList2<T, SIZE>::isFirst() const
{
	if (!isEmpty()) {
		assert(_current >= 0);
		assert(_current < _size);
		return _prev[_current] < 0;
	}
	return false;
}


template<typename T, int SIZE>
CU_HSTDEV bool DoubleLinkedList2<T, SIZE>::isLast() const
{
	if (!isEmpty()) {
		assert(_current >= 0);
		assert(_current < _size);
		return _next[_current] < 0;
	}
	return false;
}


template<typename T, int SIZE>
CU_HSTDEV bool DoubleLinkedList2<T, SIZE>::hasPrev() const
{
	if (!isEmpty()) {
		assert(_current >= 0);
		assert(_current < _size);
		return _prev[_current] >= 0;
	}
	return false;
}


template<typename T, int SIZE>
CU_HSTDEV bool DoubleLinkedList2<T, SIZE>::hasNext() const
{
	if (!isEmpty()) {
		assert(_current >= 0);
		assert(_current < _size);
		return _next[_current] >= 0;
	}
	return false;
}


template<typename T, int SIZE>
CU_HSTDEV bool DoubleLinkedList2<T, SIZE>::gotoFirst()
{
	if (!isEmpty()) {
		_current = _root;
		assert(_current >= 0);
		assert(_current < _size);
		return true;
	}
	return false;
}


template<typename T, int SIZE>
CU_HSTDEV bool DoubleLinkedList2<T, SIZE>::gotoLast()
{
	if (!isEmpty()) {
		while (gotoNext());
		return true;
	}
	return false;
}


template<typename T, int SIZE>
CU_HSTDEV bool DoubleLinkedList2<T, SIZE>::gotoNext()
{
	if (hasNext()) {
		assert(_current >= 0);
		assert(_current < _size);

		_current = _next[_current];

		assert(_current >= 0);
		assert(_current < _size);

		return true;
	}
	return false;
}


template<typename T, int SIZE>
CU_HSTDEV bool DoubleLinkedList2<T, SIZE>::gotoPrev()
{
	if (hasPrev()) {
		assert(_current >= 0);
		assert(_current < _size);

		_current = _prev[_current];

		assert(_current >= 0);
		assert(_current < _size);

		return true;
	}
	return false;
}


template<typename T, int SIZE>
CU_HSTDEV T& DoubleLinkedList2<T, SIZE>::first(T& result)
{
	if (!isEmpty()) {

		assert(_root >= 0);
		assert(_root < _size);

		_current = _root;
		result = _data[_root];
	}
	return result;
}


template<typename T, int SIZE>
CU_HSTDEV T& DoubleLinkedList2<T, SIZE>::last(T& result)
{
	if (!isEmpty()) {
		gotoLast();

		assert(_current >= 0);
		assert(_current < _size);

		result = _data[_current];
	}
	return result;
}


template<typename T, int SIZE>
CU_HSTDEV T& DoubleLinkedList2<T, SIZE>::prev(T& result)
{
	if (hasPrev()) {

		assert(_current >= 0);
		assert(_current < _size);

		_current = _prev[_current];

		assert(_current >= 0);
		assert(_current < _size);

		result = _data[_current];
	}
	return result;
}


template<typename T, int SIZE>
CU_HSTDEV T& DoubleLinkedList2<T, SIZE>::curr(T& result) const
{
	if (!isEmpty()) {

		assert(_current >= 0);
		assert(_current < _size);

		result = _data[_current];
	}
	return result;
}


template<typename T, int SIZE>
CU_HSTDEV T& DoubleLinkedList2<T, SIZE>::next(T& result)
{
	if (hasNext()) {

		assert(_current >= 0);
		if (_current >= _size) {
			assert(false);
		}
		assert(_current < _size);

		_current = _next[_current];

		assert(_current >= 0);
		assert(_current < _size);

		result = _data[_current];
	}
	return result;
}



template<typename T, int SIZE>
CU_HSTDEV void DoubleLinkedList2<T, SIZE>::insertItemBeforeCurrent(const T& newItem)
{
	if (hasPrev()) {
		addNewItem(_prev[_current], _current, newItem);
	}
	else {
		// Entweder ist die Liste noch leer oder das aktuelle Element steht in der Liste an erster Stelle.
		addNewItem(-1, _current, newItem);
	}
}


template<typename T, int SIZE>
CU_HSTDEV void DoubleLinkedList2<T, SIZE>::insertItemAfterCurrent(const T& newItem)
{
	if (hasNext()) {
		addNewItem(_current, _next[_current], newItem);
	}
	else {
		// Entweder ist die Liste noch leer oder das aktuelle Element steht in der Liste an letzter Stelle.
		addNewItem(_current, -1, newItem);
	}
}


template<typename T, int SIZE>
CU_HSTDEV void DoubleLinkedList2<T, SIZE>::append(const T& newItem)
{
	if (isEmpty()) {
		addNewItem(-1, -1, newItem);
	}
	else {
		__int16 saveCurrent = _current;
		gotoLast();
		insertItemAfterCurrent(newItem);
		_current = saveCurrent;
	}
}


template<typename T, int SIZE>
CU_HSTDEV void DoubleLinkedList2<T, SIZE>::removeCurrent()
{
	// TODO Implementieren: DoubleLinkedList2<T, SIZE>::removeCurrent()
	//if (isEmpty()) return;

	//__int16 nextIdx = _next[_current];
	//__int16 prevIdx = _prev[_current];

	//if (isFirst()) {
	//	if (hasNext()) {
	//		_prev[nextIdx] = -1;
	//		deleteItem(_current);
	//	}
	//	else {
	//		_root = -1;
	//		_current = -1;
	//		_size = 0;
	//	}
	//}
	//else if (isLast()) {
	//	_next[prevIdx] = -1;
	//	deleteItem(_current);
	//	_current = prevIdx;
	//}
	//else {
	//	_next[prevIdx] = nextIdx;
	//	_prev[nextIdx] = prevIdx;
	//	deleteItem(_current);
	//	_current = nextIdx;
	//}
}


template<typename T, int SIZE>
CU_HSTDEV bool DoubleLinkedList2<T, SIZE>::find(const T& item)
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


template<typename T, int SIZE>
CU_HSTDEV void DoubleLinkedList2<T, SIZE>::addNewItem(int prevIdx, int nextIdx, const T& newItem)
{
	if (_size >= SIZE)
	{
#ifdef __CUDACC__
		printf("DoubleLinkedList2<T, SIZE>::addNewItem(): Liste ist voll!\n");
#else
		throw std::runtime_error("DoubleLinkedList2<T, SIZE>::addNewItem(): Liste ist voll!");
#endif
	}
	else {

		int idx = _size;
		_size++;

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
}


template<typename T, int SIZE>
CU_HSTDEV void DoubleLinkedList2<T, SIZE>::deleteItem(__int16 idx)
{
	if (_size > 1) {
		__int16 nextItem = _next[idx];

		// Das zu löschende Element wird mit dem letzten Element aus _data überschrieben.
		__int16 lastItem = _size - 1;
		_data[idx] = _data[lastItem];
		_prev[idx] = _prev[lastItem];
		_next[idx] = _next[lastItem];

		// Die Verweise auf das gerade verschobene letzte Element müssen aktualisiert werden.
		// Es kann jeweils nur einen Verweis geben.
		__int16 i = 0;
		bool found = false;
		while (i < _size && !found) {
			if (_prev[i] == lastItem) {
				_prev[i] = idx;
				found = true;
			}
			i++;
		}

		i = 0;
		found = false;
		while (i < _size && !found) {
			if (_next[i] == lastItem) {
				_next[i] = idx;
				found = true;
			}
			i++;
		}

		if (nextItem == lastItem) {
			_current = idx;
		}
		else {
			_current = nextItem;
		}
	}

	_size--;

	if (_size == 1) {
		_root = _current;
	}

	if (_size == 0) {
		_root = -1;
		_current = -1;
	}
}


template<typename T, int SIZE>
CU_HSTDEV __int16 DoubleLinkedList2<T, SIZE>::getSize() const
{
	return _size;
}


