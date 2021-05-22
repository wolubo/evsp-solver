#include <algorithm>

#include "EVSP.BaseClasses/Typedefs.h"
#include "Helper.h"



	

			float toFloat(std::string s)
			{
				std::replace(s.begin(), s.end(), ',', '.');
				return std::stof(s);
			}

			std::string convert2string(float f)
			{
				std::string s = std::to_string(f);
				std::replace(s.begin(), s.end(), '.', ',');

				size_t pos = s.length() - 1;
				size_t comma = s.find(',');

				char c = s[pos];
				while (pos > 1 && pos > comma && c == '0') {
					s = s.substr(0, pos);
					pos--;
					c = s[pos];
				}
				if (c == ',') s = s.substr(0, pos);
				return s;
			}

