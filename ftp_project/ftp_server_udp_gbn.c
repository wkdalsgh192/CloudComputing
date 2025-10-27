#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/time.h>
#include <sys/ioctl.h>
#include <net/if.h>

#define MAX_PACKET_SIZE 9000

typedef struct {
    int seq_num;
    int is_eof;
    int length;
    char data[MAX_PACKET_SIZE];
} Packet;

void error(const char *msg) {
    perror(msg);
    exit(1);
}

int get_mtu(const char *ifname) {
    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) error("socket() for MTU query failed");

    struct ifreq ifr;
    strncpy(ifr.ifr_name, ifname, IFNAMSIZ - 1);
    if (ioctl(sock, SIOCGIFMTU, &ifr) < 0) {
        close(sock);
        error("ioctl(SIOCGIFMTU) failed");
    }
    close(sock);
    return ifr.ifr_mtu;
}

int main(int argc, char *argv[]) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <port> <output_file> <interface>\n", argv[0]);
        exit(1);
    }

    int sockfd, portno, n;
    socklen_t clilen;
    struct sockaddr_in serv_addr, cli_addr;

    int mtu = get_mtu(argv[3]);
    int BUF_SIZE = mtu - 28;
    if (BUF_SIZE > 9000) BUF_SIZE = 9000;
    printf("Detected MTU = %d, using UDP chunk size = %d bytes\n", mtu, BUF_SIZE);

    portno = atoi(argv[1]);
    sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0) error("ERROR opening socket");

    memset(&serv_addr, 0, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = INADDR_ANY;
    serv_addr.sin_port = htons(portno);

    if (bind(sockfd, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0)
        error("ERROR on binding");

    printf("UDP GBN server listening on port %d...\n", portno);

    FILE *fp = fopen(argv[2], "wb");
    if (!fp) error("ERROR opening output file");

    int expected_seq = 0;
    int eof_seq = -1;
    long long total_bytes = 0;
    struct timeval last_recv, now;
    gettimeofday(&last_recv, NULL);

    clilen = sizeof(cli_addr);
    Packet pkt;
    while (1) {
        n = recvfrom(sockfd, &pkt, sizeof(pkt), MSG_DONTWAIT,
                     (struct sockaddr *)&cli_addr, &clilen);
        if (n > 0) {
            gettimeofday(&last_recv, NULL);

            if (pkt.is_eof) {
                eof_seq = pkt.seq_num;
                printf("EOF received (seq=%d)\n", eof_seq);
                continue;
            }

            if (pkt.seq_num == expected_seq) {
                fwrite(pkt.data, 1, pkt.length, fp);
                total_bytes += pkt.length;
                expected_seq++;

                // ACK next expected seq (like TCP cumulative ACK)
                char ack_msg[32];
                snprintf(ack_msg, sizeof(ack_msg), "ACK %d", expected_seq);
                sendto(sockfd, ack_msg, strlen(ack_msg), 0,
                       (struct sockaddr *)&cli_addr, clilen);
            } else if (pkt.seq_num > expected_seq) {
                // NACK for the expected packet only
                char nack_msg[32];
                snprintf(nack_msg, sizeof(nack_msg), "NACK %d", expected_seq);
                sendto(sockfd, nack_msg, strlen(nack_msg), 0,
                       (struct sockaddr *)&cli_addr, clilen);
            }
        }

        gettimeofday(&now, NULL);
        if (eof_seq != -1 && expected_seq >= eof_seq)
            break;
        if (eof_seq != -1 && (now.tv_sec - last_recv.tv_sec) > 5) {
            printf("Timeout after EOF, closing.\n");
            break;
        }
    }

    fclose(fp);
    close(sockfd);
    printf("Received %lld bytes successfully.\n", total_bytes);
    return 0;
}
