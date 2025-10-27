#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <sys/ioctl.h>
#include <net/if.h>

#define MAX_PACKET_SIZE 9000
#define WINDOW_SIZE 50
#define TIMEOUT_US 500000  // 0.5s

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
    if (argc < 5) {
        fprintf(stderr, "Usage: %s <server_ip> <port> <file> <interface>\n", argv[0]);
        exit(1);
    }

    int sockfd, portno;
    struct sockaddr_in serv_addr;
    struct hostent *server;

    int mtu = get_mtu(argv[4]);
    int BUF_SIZE = mtu - 28;
    if (BUF_SIZE > 9000) BUF_SIZE = 9000;
    printf("Detected MTU = %d, chunk size = %d\n", mtu, BUF_SIZE);

    portno = atoi(argv[2]);
    sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0) error("ERROR opening socket");

    server = gethostbyname(argv[1]);
    if (!server) error("No such host");

    memset(&serv_addr, 0, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    memcpy(&serv_addr.sin_addr.s_addr, server->h_addr, server->h_length);
    serv_addr.sin_port = htons(portno);

    FILE *fp = fopen(argv[3], "rb");
    if (!fp) error("ERROR opening file");

    socklen_t servlen = sizeof(serv_addr);
    Packet window[WINDOW_SIZE];
    int base = 0, next_seq = 0, eof_sent = 0;
    long long total_bytes = 0;

    while (!eof_sent) {
        // send new packets within window
        while (next_seq < base + WINDOW_SIZE && !eof_sent) {
            Packet pkt;
            pkt.seq_num = next_seq;
            pkt.is_eof = 0;
            pkt.length = fread(pkt.data, 1, BUF_SIZE, fp);
            if (pkt.length <= 0) {
                pkt.is_eof = 1;
                pkt.length = 0;
                eof_sent = 1;
            }
            sendto(sockfd, &pkt, sizeof(int)*3 + pkt.length, 0,
                   (struct sockaddr *)&serv_addr, servlen);
            window[next_seq % WINDOW_SIZE] = pkt;
            next_seq++;
        }

        // listen for ACK/NACK
        char buf[64];
        int n = recvfrom(sockfd, buf, sizeof(buf), MSG_DONTWAIT,
                         (struct sockaddr *)&serv_addr, &servlen);
        if (n > 0) {
            int ackno;
            if (sscanf(buf, "ACK %d", &ackno) == 1) {
                if (ackno > base) base = ackno;
            } else if (sscanf(buf, "NACK %d", &ackno) == 1) {
                // resend from that seq
                for (int i = ackno; i < next_seq; i++) {
                    Packet *p = &window[i % WINDOW_SIZE];
                    sendto(sockfd, p, sizeof(int)*3 + p->length, 0,
                           (struct sockaddr *)&serv_addr, servlen);
                }
            }
        }

        // simple timeout-based resend (Go-Back-N)
        usleep(1000);
    }

    // Send EOF a few times to ensure reception
    for (int i = 0; i < 5; i++) {
        Packet eof_pkt = { .seq_num = next_seq, .is_eof = 1, .length = 0 };
        sendto(sockfd, &eof_pkt, sizeof(int)*3, 0,
               (struct sockaddr *)&serv_addr, servlen);
        usleep(10000);
    }

    fclose(fp);
    close(sockfd);
    printf("File transfer complete. Total %lld bytes sent.\n", total_bytes);
    return 0;
}
