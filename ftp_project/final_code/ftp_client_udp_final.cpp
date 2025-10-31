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
#include <unordered_set>

#include "utils.cpp"

const int SELF_PORT = 8000;
const size_t MAX_PAYLOAD = 1400;

struct Packet {
    uint32_t seq_num;
    uint32_t total_chunks;
    char data[MAX_PAYLOAD];
};

struct NACK {
    uint32_t start_seq;
    uint32_t end_seq; // inclusive
};

// Shared queue for NACKs
std::queue<uint32_t> nack_queue;
std::unordered_set<uint32_t> nack_set; // To keep track of which sequence numbers are already in the queue so we don't add duplicates
std::mutex nack_mutex;
std::condition_variable nack_cv;
std::atomic<bool> running(true);

auto start_time = std::chrono::high_resolution_clock::now();
std::streamsize file_size = 0;

// Listen for NACKs and add the missing sequence numbers to the retransmit queue
void nack_listener(int sock_fd) {
    char buffer[1024];
    while (running) {
        ssize_t n = recv(sock_fd, buffer, sizeof(buffer), 0);
        if (n <= 0) continue;

        if (n >= sizeof(uint32_t)) {
            NACK nack;
            memcpy(&nack, buffer, sizeof(NACK));
            uint32_t start_seq = ntohl(nack.start_seq);
            uint32_t end_seq = ntohl(nack.end_seq);
//            std::cout << "Received NACK for chunk " << start_seq << " to " << end_seq << "\n";

            // A NACK with missing_seq == UINT32_MAX indicates termination
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
                    if (nack_set.find(missing_seq) != nack_set.end()) continue; // Already in queue
                    nack_queue.push(missing_seq);
                    nack_set.insert(missing_seq);
                }
            }
            nack_cv.notify_one();
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <filename> <receiver_host> <receiver_port>\n";
        exit(1);
    }
    const char *filename = argv[1];

    // Get file from disk to memory
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Sender: could not open file " << filename << "\n";
        exit(1);
    }
    file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    if (file_size > (1.5 * 1024 * 1024 * 1024)) { // 1.5 GB limit
        std::cerr << "Sender: file size exceeds 1.5 GB limit\n";
        exit(1);
    }

    std::vector<char> file_buffer(file_size);
    if (!file.read(file_buffer.data(), file_size)) {
        std::cerr << "Sender: could not read file " << filename << "\n";
        exit(1);
    }

    uint32_t total_chunks = (file_size + MAX_PAYLOAD - 1) / MAX_PAYLOAD;

    // Set up a UDP connection to the receiver
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

    // Start listening for NACKs
    std::thread nack_listener_thread(nack_listener, sock_fd);

    // First, send the entire file
    start_time = std::chrono::high_resolution_clock::now();
    std::cout << "Sender: starting transmission at " << start_time.time_since_epoch().count() << "\n";
    for (uint32_t seq = 0; seq < total_chunks; ++seq) {
        Packet packet;
        packet.seq_num = htonl(seq);
        packet.total_chunks = htonl(total_chunks);
        size_t offset = seq * MAX_PAYLOAD;
        size_t chunk_size = std::min((size_t)MAX_PAYLOAD, (size_t)(file_size - offset));
        std::memcpy(packet.data, file_buffer.data() + offset, chunk_size);

        ssize_t sent = send(sock_fd, &packet, sizeof(packet.seq_num) + sizeof(packet.total_chunks) + chunk_size, 0);
        if (sent < 0) {
            std::cerr << "Sender: send() failed for chunk " << seq << "\n";
        }
    }
    std::cout << "Sender: initial transmission complete, sent " << total_chunks << " chunks\n";
    // Keep sending the last chunk every 200ms to make sure that the last chunk is received,
    // so that the receiver can detect gaps at the end of the file. If the last chunk is not
    // received, the receiver doesn't know if it was lost or never sent, and therefore cannot
    // send NACKs for it.

    // Resend lost chunks
    while (running) {
        std::unique_lock<std::mutex> lock(nack_mutex);
        nack_cv.wait(lock, []{ return !nack_queue.empty() || !running;});

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
                size_t chunk_size = std::min((size_t)MAX_PAYLOAD, (size_t)(file_size - offset));
                std::memcpy(packet.data, file_buffer.data() + offset, chunk_size);

                ssize_t sent = send(sock_fd, &packet, sizeof(packet.seq_num) + sizeof(packet.total_chunks) + chunk_size, 0);
//                std::cout << "Resending chunk " << missing_seq << "\n";
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

    // Stop once FIN is received from the server (i.e. a NACK with seq_num == -1)
    freeaddrinfo(serv_info);
    running = false;
    close(sock_fd);
    nack_listener_thread.join();
    return 0;
}