#include <iostream>
#include <iomanip>
#include <cstdio>
#include <cstdlib>

const char* gray = "\033[0;37;90m";
const char* reset = "\033[0m";
const char* red = "\033[0;31;40m";
const char* green="\033[0;32;40m";

#define LOG(x) do {\
    std::cout << std::left << std::setw(40) << x << gray \
            << "[func:" << std::setw(20)<< std::setfill('.') << __FUNCTION__ \
            << " file:" << std::setw(15)<< std::setfill('.') << std::string(__FILE__).substr(std::string(__FILE__).find_last_of("/\\")+1) \
            << " line:" << std::setw(3) << std::setfill('.') << __LINE__ << "]" << reset << std::endl << std::setfill(' '); \
} while(0) 
// #define DEBUG(fmt, args...) printf("\033[31m[TEST: %s:%d:%s:%s] "#fmt"\033[0m\r\n", __FUNCTION__, __LINE__, __DATE__, __TIME__, ##args)
#define INFO(fmt, args...) printf("%s[INFO: %s:#%d] "#fmt"%s\r\n", gray, __FUNCTION__, __LINE__, ##args, reset)
#define ERROR(fmt, args...) printf("%s[ERROR: %s:%d] "#fmt"%s\r\n", red, __FUNCTION__, __LINE__, ##args, reset)
#define FUNC_ENTER printf("%s>>>> function %s ------------------------------------- %s\r\n", gray, __FUNCTION__, reset)
#define FUNC_EXIT  printf("%s<<<< function %s ------------------------------------- %s\r\n", gray, __FUNCTION__, reset)