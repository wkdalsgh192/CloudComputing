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
const int NACK_INTERVAL_MS = 5;

struct PacketHeader {
    uint32_t seq_num;
    uint32_t total_chunks;
};

struct NACK {
    uint32_t start_seq;
    uint32_t end_seq;
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

void nack_thread(int sock_fd, sockaddr_storage sender_addr, socklen_t sender_len) {
    while (running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(NACK_INTERVAL_MS));

        if (total_chunks == 0) continue;

        std::vector<uint32_t> missing;
        {
            std::lock_guard<std::mutex> lock(recv_mutex);
            for (uint32_t i = 0; i < max_chunk_received; ++i) {
                if (!received_chunks[i]) {
                    missing.push_back(i);
                }
            }
        }
        std::cout << "Received " << n_chunks_received << " chunks. Missing " << missing.size() << " chunks. Max received chunk is " << max_chunk_received << "\n";

        // find NACK ranges
        std::vector<NACK> nack_ranges;
        uint32_t range_start = UINT32_MAX;
        for (size_t i = 0; i < missing.size(); ++i) {
            if (range_start == UINT32_MAX) {
                range_start = missing[i];
            }
            // group consecutive losses
            if (i == missing.size() - 1 || missing[i] + 1 != missing[i+1]) {
                nack_ranges.push_back({range_start, missing[i]});
                range_start = UINT32_MAX;
            }
        }

        // send all nack packets to client
        for (const auto& nack : nack_ranges) {
            NACK nack_to_send = nack;
            nack_to_send.start_seq = htonl(nack.start_seq);
            nack_to_send.end_seq = htonl(nack.end_seq);
            ssize_t sent = sendto(sock_fd, &nack_to_send, sizeof(nack_to_send), 0, (struct sockaddr *)&sender_addr, sender_len);
            if (sent < 0) {
                std::cerr << "Receiver: sendto() failed for NACK of range " << nack.start_seq << "-" << nack.end_seq << "\n";
            }
        }

        if (nack_ranges.empty()) {
            // check if all chunks have been received
            bool all_received = true;
            {
                std::lock_guard<std::mutex> lock(recv_mutex);
                for (bool r : received_chunks) {
                    if (!r) {
                        all_received = false;
                        break;
                    }
                }
            }
            if (all_received) {
                std::cout << "Receiver: all chunks received, sending FIN and terminating\n";
                NACK fin_ack;
                fin_ack.start_seq = htonl(UINT32_MAX);
                fin_ack.end_seq = htonl(UINT32_MAX);
                sendto(sock_fd, &fin_ack, sizeof(fin_ack), 0, (struct sockaddr *)&sender_addr, sender_len);
                running = false;

                // write all received data to a file
                std::ofstream outfile("received_file", std::ios::binary);
                for (const auto& chunk : file_data) {
                    outfile.write(chunk.data(), chunk.size());
                }
                outfile.close();
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


    // Peek at the first packet to get sender's address
    // you'll send NACKs back to that sender
    char tmp_buff[1];
    sockaddr_storage sender_addr{};
    socklen_t sender_len = sizeof(sender_addr);
    recvfrom(sock_fd, tmp_buff, sizeof(tmp_buff), MSG_PEEK, (struct sockaddr *) &sender_addr, &sender_len); // preserve all packets with MSG_PEEK

    // Start receiving packets 
    std::thread recv_thread_worker(receiver_thread, sock_fd);

    // Start sending nack
    std::thread nack_thread_worder(nack_thread, sock_fd, sender_addr, sender_len);

    recv_thread_worker.join();
    nack_thread_worder.join();

    // Terminate connection
    close(sock_fd);
    return 0;
}