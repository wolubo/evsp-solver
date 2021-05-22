#pragma once

//#include <limits.h>

#include "SafeType.hpp"
#include "SafeFloatType.hpp"

typedef unsigned short ushort;

struct EdgeIdType {};
typedef SafeType<EdgeIdType, short, -1> EdgeId;

struct NodeIdType {};
typedef SafeType<NodeIdType, short, -1> NodeId;

struct EmptyTripIdType {};
typedef SafeType<EmptyTripIdType, short, -1> EmptyTripId;

struct ServiceTripIdType {};
typedef SafeType<ServiceTripIdType, short, -1> ServiceTripId;

struct StopIdType {};
typedef SafeType<StopIdType, short, -1> StopId;

struct DepotIdType {};
typedef SafeType<DepotIdType, short, -1> DepotId;

struct ChargingStationIdType {};
typedef SafeType<ChargingStationIdType, short, -1> ChargingStationId;

struct VehicleTypeIdType {};
typedef SafeType<VehicleTypeIdType, short, -1> VehicleTypeId;

struct CirculationIdType {};
typedef SafeType<CirculationIdType, short, -1> CirculationId;

struct CircStepIndexType {};
typedef SafeType<CircStepIndexType, short, -1> CircStepIndex;

struct VehicleTypeGroupIdType {};
typedef SafeType<VehicleTypeGroupIdType, short, -1> VehicleTypeGroupId;

struct RouteIdType {};
typedef SafeType<RouteIdType, short, -1> RouteId;

struct ItemHandleType {};
typedef SafeType<ItemHandleType, short, -1> ItemHandle;

struct DistanceInMetersType {};
typedef SafeType<DistanceInMetersType, int, INT_MIN> DistanceInMeters;

struct DurationInSecondsType {};
typedef SafeType<DurationInSecondsType, int, INT_MIN> DurationInSeconds;

struct AmountOfMoneyType {};
typedef SafeType<AmountOfMoneyType, int, INT_MIN> AmountOfMoney;

struct KilowattHourType {};
typedef SafeFloatType<KilowattHourType> KilowattHour;


