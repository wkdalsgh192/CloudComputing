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

const int SELF_PORT = 8000;
const size_t MAX_PAYLOAD = 1400;

struct PacketHeader {
    uint32_t seq_num;
    uint32_t total_chunks;
};

std::vector<bool> received_chunks;
std::vector<std::vector<char>> file_data;
std::atomic<uint32_t> total_chunks{0};
std::atomic<u_int32_t> max_chunk_received{0};
std::atomic<uint32_t> n_chunks_received{0};
std::mutex recv_mutex;
std::atomic<bool> running(true);

void receiver_thread(int sock_fd) {
    char buffer[sizeof(PacketHeader) + MAX_PAYLOAD];
    while (running) {
        ssize_t n = recv(sock_fd, buffer, sizeof(buffer), 0);
        if (n < (ssize_t) sizeof(PacketHeader)) continue;

        PacketHeader header;
        memcpy(&header, buffer, sizeof(PacketHeader));
        uint32_t seq = ntohl(header.seq_num);
        uint32_t total = ntohl(header.total_chunks);

        // parse the original file data from the first packet
        if (total_chunks == 0) {
            total_chunks = total;
            received_chunks.resize(total, false);
            file_data.resize(total);
        }

        size_t chunk_size = n - sizeof(PacketHeader);
        {
            std::lock_guard<std::mutex> lock(recv_mutex);
            if (seq < total_chunks && !received_chunks[seq]) {
                received_chunks[seq] = true;
                if (seq > max_chunk_received) {
                    max_chunk_received = seq;
                }
                n_chunks_received++;
                file_data[seq].assign(buffer + sizeof(PacketHeader), buffer + sizeof(PacketHeader) + chunk_size);
            }
        }
    }
}

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

    // Peek at the first packet to get sender's address
    // you'll send NACKs back to that sender

    // Start receiving packets 
    std::thread recv_thread_worker(receiver_thread, sock_fd);

    recv_thread_worker.join();

    // Terminate connection
    close(sock_fd);
    return 0;
}