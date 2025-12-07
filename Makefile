CXX = g++
CXXFLAGS = -std=c++11 -pthread -Wall -Wextra
TARGET = neural_network
SOURCE = main.cpp

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCE)

clean:
	rm -f $(TARGET) output.txt

.PHONY: all clean

