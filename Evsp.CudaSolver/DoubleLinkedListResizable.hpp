#pragma once

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda.h> 
#endif

#include <stdexcept>
#include <assert.h>
#include "EVSP.BaseClasses/DeviceLaunchParameters.h"
#include "CudaCheck.h"


/// <summary>
/// Doppelt verkettete Liste beliebiger Objekte. Die Liste verwaltet einen internen Zeiger auf das gerade aktuelle
/// Objekt. Dieser Zeiger kann mit Methoden wie first() oder next() durch die Liste bewegt werden.
///
/// Die Klasse ist nicht Thread-Safe! Sie ist für die Verwendung durch einen einzelnen Thread bestimmt.
///
/// Intern werden Arrays verwendet, um die Objekte zu speichern und zu verwalten. Diese Arrays haben eine bei der
/// Instantiierung definierte Größe. Sollten der Liste mehr Objekte hinzugefügt werden, als die internen Arrays
/// fassen können, so werden sie automatisch vergrößert. Dies ist jedoch eine sehr kostenintensive Operation,
/// die nach Möglichkeit vermieden werden sollte.
/// </summary>
template<typename T>
class DoubleLinkedListResizable
{
public:

	/// <summary>
	/// Kopiert eine Liste aus dem Device- in den Host-Speicher.
	/// Falls die Liste Pointer enthält, so zeigen diese nach dem Kopieren immer noch in den Device-Speicher. Die einzelnen Objekte
	/// müssen dann in den Host-Speicher kopiert und in einer neuen Liste abgelegt werden.
	/// </summary>
	/// <returns>Zeiger auf die Kopie der Liste im Host-Speicher</returns>
	static DoubleLinkedListResizable<T>* cpy2hst(const DoubleLinkedListResizable<T> *devicePtr);

	/// <summary>
	/// Kopiert eine Liste aus dem Device- in den Host-Speicher.
	/// Für jedes Element wird dabei die Funktion 'translate()' aufgerufen. Auf diese Weise können bspw. Listen kopiert werden, 
	/// welche Pointer enthalten.
	/// </summary>
	/// <returns>Zeiger auf die Kopie der Liste im Host-Speicher</returns>
	static DoubleLinkedListResizable<T>* cpy2hst(const DoubleLinkedListResizable<T> *devicePtr, T translate(T));

	/// <summary>
	/// Lösche eine Liste aus dem Device-Speicher. Sofern die List Pointer enthält müssen die entsprechenden Objekte vorab 
	/// gelöscht werden (Liste auf den Host kopieren und dann wbCudaFree für jedes Objekt in der Liste).
	/// </summary>
	static void delOnDev(DoubleLinkedListResizable<T> *devicePtr);

	/// <summary>
	/// Erzeugt eine leere Liste.
	/// </summary>
	/// <param name="initialCapacity">Initiale Kapazität der Liste.</param>
	/// <param name="growRate">Anzahl der Elemente, um welche die Kapazität bei Bedarf erhöht wird.</param>
	CU_HSTDEV DoubleLinkedListResizable(__int16 initialCapacity, __int16 growRate);

	CU_HSTDEV DoubleLinkedListResizable(const DoubleLinkedListResizable& other) { assign(false); }

	/// <summary>
	/// Destruktor.
	/// </summary>
	CU_HSTDEV ~DoubleLinkedListResizable();

	DoubleLinkedListResizable<T>& operator=(const DoubleLinkedListResizable<T> &rhs) { assert(false); return *this; }

	/// <summary>
	/// Liefert einen Zeiger auf eine Kopie des Objekts im Device-Speicher. Jeder Aufruf generiert eine neue Kopie.
	/// Falls die Liste Pointer enthält müssen die entsprechenden Objekte vorab einzeln auf das Device kopiert und in 
	/// einer neuen Liste abgelegt werden.
	/// </summary>
	/// <returns>Zeiger auf die Kopie der Liste im Device-Speicher</returns>
	DoubleLinkedListResizable<T>* cpy2dev();

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

	/// <summary>
	/// Liefert die aktuelle Kapazität der Liste. 
	/// </summary>
	/// <returns>Kapazität der Liste.</returns>
	CU_HSTDEV __int16 getCapacity() const;

private:
	DoubleLinkedListResizable() {}
	CU_HSTDEV void addNewItem(int prevIdx, int nextIdx, const T& newItem);
	CU_HSTDEV void deleteItem(__int16 idx);
	CU_HSTDEV void grow();

	T* _data;
	__int16 *_prev, *_next;
	__int16 _root;
	__int16 _current;
	__int16 _capacity;
	__int16 _growRate;
	__int16 _size;			// Aktuelle Anzahl der Elemente in der Liste.
};


#ifdef __CUDACC__
template<typename T>
DoubleLinkedListResizable<T>* DoubleLinkedListResizable<T>::cpy2hst(const DoubleLinkedListResizable<T> *devicePtr)
{
	DoubleLinkedListResizable<T>* retVal = 0;
	if (devicePtr) {
		retVal = new DoubleLinkedListResizable<T>();
		CUDA_CHECK(cudaMemcpy(retVal, devicePtr, sizeof(DoubleLinkedListResizable<T>), cudaMemcpyDeviceToHost));

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
template<typename T>
DoubleLinkedListResizable<T>* DoubleLinkedListResizable<T>::cpy2hst(const DoubleLinkedListResizable<T> *devicePtr, T translate(T))
{
	DoubleLinkedListResizable<T>* list = cpy2hst(devicePtr);
	DoubleLinkedListResizable<T>* retVal = new DoubleLinkedListResizable<T>(list->_capacity, list->_growRate);
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
template<typename T>
void DoubleLinkedListResizable<T>::delOnDev(DoubleLinkedListResizable<T> *devicePtr)
{
	if (devicePtr) {
		DoubleLinkedListResizable<T> temp;
		CUDA_CHECK(cudaMemcpy(&temp, devicePtr, sizeof(DoubleLinkedListResizable<T>), cudaMemcpyDeviceToHost));

		if (temp._prev) CUDA_CHECK(wbCudaFree(__FILE__, __LINE__, temp._prev));
		if (temp._next) CUDA_CHECK(wbCudaFree(__FILE__, __LINE__, temp._next));
		if (temp._data) CUDA_CHECK(wbCudaFree(__FILE__, __LINE__, temp._data));
		CUDA_CHECK(wbCudaFree(__FILE__, __LINE__, devicePtr));
	}
}
#endif


template<typename T>
CU_HSTDEV DoubleLinkedListResizable<T>::DoubleLinkedListResizable(__int16 initialCapacity, __int16 growRate)
	: _capacity(initialCapacity), _growRate(growRate), _root(-1), _current(-1), _size(0)
{
	assert(initialCapacity > 0);
	assert(growRate > 0);

	_data = new T[_capacity];
	assert(_data);

	_prev = new __int16[_capacity];
	assert(_prev);

	_next = new __int16[_capacity];
	assert(_next);

	for (__int16 i = 0; i < _capacity; i++) {
		_prev[i] = -1;
		_next[i] = -1;
	}
}


template<typename T>
CU_HSTDEV DoubleLinkedListResizable<T>::~DoubleLinkedListResizable()
{
	delete[] _data;
	delete[] _prev;
	delete[] _next;
}


#ifdef __CUDACC__
template<typename T>
DoubleLinkedListResizable<T>* DoubleLinkedListResizable<T>::cpy2dev()
{
	DoubleLinkedListResizable<T> *retVal;

	CUDA_CHECK(wbCudaMalloc(__FILE__, __LINE__, (void**)&retVal, sizeof(DoubleLinkedListResizable<T>)));
	
	CUDA_CHECK(cudaMemcpy(retVal, this, sizeof(DoubleLinkedListResizable<T>), cudaMemcpyHostToDevice));

	CUDA_CHECK(wbCudaMalloc(__FILE__, __LINE__, (void**)&retVal->_prev, _capacity * sizeof(__int16)));
	
	CUDA_CHECK(cudaMemcpy(retVal->_prev, _prev, _size * sizeof(__int16), cudaMemcpyHostToDevice));

	CUDA_CHECK(wbCudaMalloc(__FILE__, __LINE__, (void**)&retVal->_next, _capacity * sizeof(__int16)));
	
	CUDA_CHECK(cudaMemcpy(retVal->_next, _next, _size * sizeof(__int16), cudaMemcpyHostToDevice));

	CUDA_CHECK(wbCudaMalloc(__FILE__, __LINE__, (void**)&retVal->_data, _capacity * sizeof(T)));
	
	CUDA_CHECK(cudaMemcpy(retVal->_data, _data, _size * sizeof(T), cudaMemcpyHostToDevice));

	return retVal;
}
#endif


template<typename T>
CU_HSTDEV bool DoubleLinkedListResizable<T>::isEmpty() const
{
	return _root < 0;
}


template<typename T>
CU_HSTDEV bool DoubleLinkedListResizable<T>::isFirst() const
{
	if (!isEmpty()) {
		assert(_current >= 0);
		
		return _prev[_current] < 0;
	}
	return false;
}


template<typename T>
CU_HSTDEV bool DoubleLinkedListResizable<T>::isLast() const
{
	if (!isEmpty()) {
		assert(_current >= 0);
		
		return _next[_current] < 0;
	}
	return false;
}


template<typename T>
CU_HSTDEV bool DoubleLinkedListResizable<T>::hasPrev() const
{
	if (!isEmpty()) {
		assert(_current >= 0);
		
		return _prev[_current] >= 0;
	}
	return false;
}


template<typename T>
CU_HSTDEV bool DoubleLinkedListResizable<T>::hasNext() const
{
	if (!isEmpty()) {
		assert(_current >= 0);
		return _next[_current] >= 0;
	}
	return false;
}


template<typename T>
CU_HSTDEV bool DoubleLinkedListResizable<T>::gotoFirst()
{
	if (!isEmpty()) {
		_current = _root;
		assert(_current >= 0);
		return true;
	}
	return false;
}


template<typename T>
CU_HSTDEV bool DoubleLinkedListResizable<T>::gotoLast()
{
	if (!isEmpty()) {
		while (gotoNext());
		return true;
	}
	return false;
}


template<typename T>
CU_HSTDEV bool DoubleLinkedListResizable<T>::gotoNext()
{
	if (hasNext()) {
		assert(_current >= 0);

		_current = _next[_current];

		assert(_current >= 0);

		return true;
	}
	return false;
}


template<typename T>
CU_HSTDEV bool DoubleLinkedListResizable<T>::gotoPrev()
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
CU_HSTDEV T& DoubleLinkedListResizable<T>::first(T& result)
{
	if (!isEmpty()) {

		assert(_root >= 0);
		//assert(_root < _size);

		_current = _root;
		result = _data[_root];
	}
	return result;
}


template<typename T>
CU_HSTDEV T& DoubleLinkedListResizable<T>::last(T& result)
{
	if (!isEmpty()) {
		gotoLast();

		assert(_current >= 0);
		

		result = _data[_current];
	}
	return result;
}


template<typename T>
CU_HSTDEV T& DoubleLinkedListResizable<T>::prev(T& result)
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
CU_HSTDEV T& DoubleLinkedListResizable<T>::curr(T& result) const
{
	if (!isEmpty()) {

		assert(_current >= 0);
		

		result = _data[_current];
	}
	return result;
}


template<typename T>
CU_HSTDEV T& DoubleLinkedListResizable<T>::next(T& result)
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
CU_HSTDEV void DoubleLinkedListResizable<T>::insertItemBeforeCurrent(const T& newItem)
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
CU_HSTDEV void DoubleLinkedListResizable<T>::insertItemAfterCurrent(const T& newItem)
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
CU_HSTDEV void DoubleLinkedListResizable<T>::append(const T& newItem)
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


template<typename T>
CU_HSTDEV void DoubleLinkedListResizable<T>::removeCurrent()
{
	if (isEmpty()) return;

	__int16 nextIdx = _next[_current];
	__int16 prevIdx = _prev[_current];

	if (isFirst()) {
		if (hasNext()) {
			_prev[nextIdx] = -1;
			deleteItem(_current);
			_root = nextIdx;
			_current = nextIdx;
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
CU_HSTDEV bool DoubleLinkedListResizable<T>::find(const T& item)
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
CU_HSTDEV void DoubleLinkedListResizable<T>::addNewItem(int prevIdx, int nextIdx, const T& newItem)
{
	if (_size >= _capacity) grow();
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


template<typename T>
CU_HSTDEV void DoubleLinkedListResizable<T>::grow()
{
	assert(_size >= _capacity);
	assert(_data);
	assert(_prev);
	assert(_next);

	_capacity += _growRate;
	assert(_size <= _capacity);

	T *newData = new T[_capacity];
	__int16 *newPrev = new __int16[_capacity];
	__int16 *newNext = new __int16[_capacity];

	assert(newData);
	assert(newPrev);
	assert(newNext);

	//for (int i = 0; i < _size; i++) {
	//	newData[i] = _data[i];
	//	newPrev[i] = _prev[i];
	//	newNext[i] = _next[i];
	//}

	memcpy(newData, _data, _size * sizeof(T));
	memcpy(newPrev, _prev, _size * sizeof(__int16));
	memcpy(newNext, _next, _size * sizeof(__int16));

	delete[] _data;
	delete[] _prev;
	delete[] _next;

	_data = newData;
	_prev = newPrev;
	_next = newNext;
}


template<typename T>
CU_HSTDEV void DoubleLinkedListResizable<T>::deleteItem(__int16 idx)
{
	// Das zu löschende Element wird mit dem letzten Element aus _data überschrieben.
	_data[idx] = _data[_size];
	_prev[idx] = _prev[_size];
	_next[idx] = _next[_size];

	// Die Verweise auf das gerade verschobene letzte Element müssen aktualisiert werden.
	// Es kann jeweils nur einen Verweis geben.
	__int16 i = 0;
	bool found = false;
	while (i < _size && !found) {
		if (_prev[i] == _size) {
			_prev[i] = idx;
			found = true;
		}
		i++;
	}

	i = 0;
	found = false;
	while (i < _size && !found) {
		if (_next[i] == _size) {
			_next[i] = idx;
			found = true;
		}
		i++;
	}

	_size--;
}


template<typename T>
CU_HSTDEV __int16 DoubleLinkedListResizable<T>::getSize() const
{
	return _size;
}


template<typename T>
CU_HSTDEV __int16 DoubleLinkedListResizable<T>::getCapacity() const
{
	return _capacity;
}


