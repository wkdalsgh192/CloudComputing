#include <iostream>
#include <cstring>
#include <fstream>
#include <vector>
#include <chrono>
#include <atomic>
#include <queue>
#include <mutex>
#include <thread>
#include <unordered_set>
#include <sys/socket.h>
#include <condition_variable>

#include "utils.cpp"

const size_t MAX_PAYLOAD = 1400;

auto start_time = std::chrono::high_resolution_clock::now();
std::streamsize file_size = 0;
std::queue<uint32_t> nack_queue;
std::unordered_set<uint32_t> nack_set;
std::mutex nack_mutex;
std::condition_variable nack_cv;
std::atomic<bool> running(true);

struct Packet {
    uint32_t seq_num;
    uint32_t total_chunks;
    char data[MAX_PAYLOAD];
};

struct NACK {
    uint32_t start_seq;
    uint32_t end_seq;
};

void nack_listener(int sock_fd) {
    char buffer[1024];
    while (running) {
        ssize_t n = recv(sock_fd, buffer, sizeof(buffer), 0);

        // sanity check -- at least 4 bytes
        if (n >= sizeof(uint32_t)) {
            NACK nack;
            memcpy(&nack, buffer, sizeof(NACK));
            uint32_t start_seq = ntohl(nack.start_seq);
            uint32_t end_seq = ntohl(nack.end_seq);

            if (start_seq == UINT32_MAX) {
                running = false;
                nack_cv.notify_all();
                std::cout << "Received FIN signal, terminating sender\n";
                auto end_time = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
                std::cout << "Sender: ending transmission at " << end_time.time_since_epoch().count() << "\n";
                std::cout << "Total time taken: " << duration << " ms\n";
                std::cout << "Total file size transferred: " << file_size << " bytes\n";
                std::cout << "Effective throughput: " << (file_size * 8) / (duration / 1000.0) / 1e6 << " Mbps\n";
                break;
            }

            {
                std::lock_guard<std::mutex> lock(nack_mutex);
                for (uint32_t missing_seq = start_seq; missing_seq <= end_seq; ++missing_seq) {
                    if (nack_set.find(missing_seq) != nack_set.end()) continue; // already in queue
                    nack_queue.push(missing_seq);
                    nack_set.insert(missing_seq);
                }
            }
            nack_cv.notify_one();
        }
    }
}

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

    std::thread nack_listener_thread(nack_listener, sock_fd);
    
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
    
    while (running) {
        std::unique_lock<std::mutex> lock(nack_mutex);
        nack_cv.wait(lock, []{ return !nack_queue.empty() || !running; });

        while (!nack_queue.empty() && running) {
            uint32_t missing_seq = nack_queue.front();

            nack_queue.pop();
            nack_set.erase(missing_seq);
            lock.unlock();

            if (missing_seq < total_chunks) {
                Packet packet;
                packet.seq_num = htonl(missing_seq);
                packet.total_chunks = htonl(total_chunks);
                size_t offset = missing_seq * MAX_PAYLOAD;
                size_t chunk_size = std::min((size_t) MAX_PAYLOAD, (size_t) (file_size - offset));
                std::memcpy(packet.data, file_buffer.data() + offset, chunk_size);

                ssize_t sent = send(sock_fd, &packet, sizeof(packet.seq_num) + sizeof(packet.total_chunks) + chunk_size, 0);
                if (sent < 0) {
                    std::cerr << "Sender: resend() failed for chunk " << missing_seq << "\n";
                }
            }
            if (missing_seq == UINT32_MAX) {
                running = false;
                break;
            }
            lock.lock();
        }
    }
    // stop once FIN is received from the server
    freeaddrinfo(serv_info);
    running = false;
    close(sock_fd);
    return 0;
}