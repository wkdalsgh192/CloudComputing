#include <iostream>
#include <cstring>
#include <fstream>
#include <vector>
#include <chrono>
#include <atomic>
#include <queue>
#include <thread>
#include <sys/socket.h>
#include <condition_variable>

#include "utils.cpp"

const size_t MAX_PAYLOAD = 1400;

auto start_time = std::chrono::high_resolution_clock::now();
std::streamsize file_size = 0;
std::atomic<bool> running(true);

struct Packet {
    uint32_t seq_num;
    uint32_t total_chunks;
    char data[MAX_PAYLOAD];
};

int main(int argc, char *argv[]) {

    // check args
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <filename> <receiver_host> <receiver_port>\n";
        exit(1);
    }
    const char *filename = argv[1];

    // get file from disk to memory
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Sender: could not open file " << filename << "\n";
        exit(1);
    }
    file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> file_buffer(file_size);
    if (!file.read(file_buffer.data(), file_size)) {
        std::cerr << "Sender: could not read file " << filename << "\n";
        exit(1);
    }

    uint32_t total_chunks = (file_size + MAX_PAYLOAD - 1) / MAX_PAYLOAD; // why MAX_PAYLOAD is added?
    
    // set up a UDP connection to the receiver
    const char *receiver_host = argv[2];
    const char *receiver_port = argv[3];

    struct addrinfo hints = get_hints(false);
    struct addrinfo *serv_info;
    if (getaddrinfo(receiver_host, receiver_port, &hints, &serv_info) != 0) {
        std::cerr << "Sender: getaddrinfo() failed\n";
        exit(1);
    }   

    int sock_fd = connect_first(serv_info);
    if (sock_fd == -1) {
        std::cerr << "Sender: connect_first() failed\n";
        exit(1);
    }
    
    // send the entire file
    start_time = std::chrono::high_resolution_clock::now();
    for (uint32_t seq = 0; seq < total_chunks; ++seq) {
        // generate a packet
        Packet packet;
        packet.seq_num = htonl(seq);
        packet.total_chunks = htonl(total_chunks);
        size_t offset = seq * MAX_PAYLOAD;
        size_t chunk_size = std::min((size_t) MAX_PAYLOAD, (size_t)(file_size - offset));
        std::memcpy(packet.data, file_buffer.data() + offset, chunk_size);

        ssize_t sent = send(sock_fd, &packet, sizeof(packet.seq_num) + sizeof(packet.total_chunks) + chunk_size, 0);
        if (sent < 0) {
            std::cerr << "Sender: send() failed for chunk " << seq << "\n";
        }
    }
    std::cout << "Sender: initial transmission complete, sent " << total_chunks << " chunks\n";
    
    // stop once FIN is received from the server
    freeaddrinfo(serv_info);
    running = false;
    close(sock_fd);
    return 0;
}