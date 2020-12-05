cmake_minimum_required(VERSION 3.5)

execute_process(COMMAND python3.5-config --cflags OUTPUT_VARIABLE PY_C_FLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND python3.5-config --ldflags OUTPUT_VARIABLE PY_LD_FLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-O0 -g -Wall ${PY_C_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${PY_LD_FLAGS}")

link_directories(/usr/lib/python3.5/config-3.5m-x86_64-linux-gnu/)

add_library(cegar_net ${CMAKE_CURRENT_SOURCE_DIR}/src/network.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/powerset.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/interval.cpp)
target_include_directories(cegar_net PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../eigen)

add_executable(learn ${CMAKE_CURRENT_SOURCE_DIR}/src/learn.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/strategy.cpp)
target_include_directories(learn PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../eigen)
target_include_directories(learn PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../elina/elina_zonotope)
target_link_libraries(learn cegar_net bayesopt nlopt gmp mpfr zonotope elinaux python3.5 pthread dl util m gomp)

add_executable(run ${CMAKE_CURRENT_SOURCE_DIR}/src/run_charon.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/strategy.cpp)
target_include_directories(run PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../eigen)
target_link_libraries(run cegar_net gmp mpfr zonotope elinaux python3.5 pthread dl util m gomp)

set(CMAKE_CXX_COMPILER mpicxx)
add_definitions(-DCHARON_HOME="${CMAKE_CURRENT_SOURCE_DIR}/")
