#include <iostream>
#include <cstring>
#include <sys/socket.h>
#include <netdb.h>
#include <fstream>
#include <queue>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <arpa/inet.h>
#include <cstdint>

#include "utils.cpp"

int main() {
    
    // Listen for UDP connection
    struct addrinfo hints = get_hints(false);
    struct addrinfo *serv_info;
    if (getaddrinfo(nullptr, std::to_string(SELF_PORT).c_str(), &hints, &serv_info) != 0) {
        std::cerr << "Receiver: getaddrinfo() failed\n";
        exit(1);
    }

    int sock_fd = bind_first(serv_info, false);
    if (sock_fd == -1) {
        std::cerr << "Receiver: bind_first() failed\n";
        exit(1);
    }

    sockaddr_storage sender_addr{};
    socklen_t sender_len = sizeof(sender_addr);

    // Start receiving packets 

    // Terminate connection
    close(sock_fd);
    return 0;
}