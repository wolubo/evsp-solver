//
//#include "ChargingSystem.h"
//
//using namespace std;
////using namespace boost;
//
//
//
//	
//		namespace Model {
//
//
//			ChargingSystem::ChargingSystem(const string &legacyId, const string &theName)
//				: _legacyId(legacyId), _name(theName)
//			{
//				if (theName.length() == 0) throw  invalid_argument("theName");
//			}
//
//
//			std::string ChargingSystem::toString()
//			{
//				return "ChargingSystem: Name=" + _name;
//			}
//
//
//#ifdef _DEBUG
//			void ChargingSystem::write2file(std::ofstream &txtfile)
//			{
//				txtfile << _legacyId << ";" << _name << endl;
//			}
//#endif
//
//		}
//	}
//}