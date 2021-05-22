//#pragma once
//
//
//#include <string>
//#include <map>
//#include <vector>
//#include <iostream>
//#include <fstream>
//
//
//	
//		namespace Model {
//
//			/// <summary>
//			/// Repräsentiert ein Aufladesystem.
//			/// </summary>
//			class ChargingSystem
//			{
//
//			public:
//
//				ChargingSystem() = delete;
//				~ChargingSystem() {}
//
//
//					/// <summary>
//					/// Erzeugt ein Aufladesystem
//					/// </summary>
//					/// <param name="theName">Bezeichnung des Aufladesystems</param>
//				ChargingSystem(const std::string& legacyId, const std::string &theName);
//
//				std::string toString();
//
//				int getId() const { return _id; }
//
//#ifdef _DEBUG
//				void write2file(std::ofstream &txtfile);
//				std::string getLegacyId() { return _legacyId; }
//#endif
//
//#pragma region private_member_variables
//			private:
//
//				int _id;
//
//				/// <summary>
//				/// Bezeichnung des Aufladesystems
//				/// </summary>
//				std::string _name;
//
//				/// <summary>
//				/// Liste der vom Aufladesystem unterstützten Fahrzeugtypen.
//				/// </summary>
//				//public ImmutableHashSet<VehicleType> SupportetVehicleTypes { get { return _supportetVehicleTypes.ToImmutableHashSet(); } }
//				//protected HashSet<VehicleType> _supportetVehicleTypes = new HashSet<VehicleType>();
//
//#ifdef _DEBUG
//				std::string _legacyId;
//#endif
//
//#pragma endregion
//			};
//		}
//	}
//}