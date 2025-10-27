#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <sys/time.h>
#include <sys/ioctl.h>
#include <net/if.h>

#define MAX_PACKET_SIZE 9000
#define HEADER_SIZE (sizeof(int) + sizeof(int) + sizeof(int))

typedef struct {
    int seq_num;   // sequence number
    int is_eof;    // 1 if EOF marker
    int length;    // number of bytes in data[]
    char data[MAX_PACKET_SIZE];
} Packet;

void error(const char *msg) {
    perror(msg);
    exit(1);
}

// get MTU from an interface (e.g., "eth0")
int get_mtu(const char *ifname) {
    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) error("socket() for MTU query failed");

    struct ifreq ifr;
    strncpy(ifr.ifr_name, ifname, IFNAMSIZ-1);

    if (ioctl(sock, SIOCGIFMTU, &ifr) < 0) {
        close(sock);
        error("ioctl(SIOCGIFMTU) failed");
    }
    close(sock);
    return ifr.ifr_mtu;
}

int main(int argc, char *argv[]) {
    if (argc < 5) {
        fprintf(stderr,"Usage: %s <server_ip> <port> <file_to_send> <interface>\n", argv[0]);
        exit(1);
    }

    int sockfd, portno, n;
    struct sockaddr_in serv_addr;
    struct hostent *server;

    // get MTU and calculate safe payload
    int mtu = get_mtu(argv[4]);
    int BUF_SIZE = mtu - 28;   // MTU minus IP+UDP headers
    if (BUF_SIZE > 9000) BUF_SIZE = 9000; // safety cap

    printf("Detected MTU = %d, using UDP chunk size = %d bytes\n", mtu, BUF_SIZE);

    portno = atoi(argv[2]);
    sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0) error("ERROR opening socket");

    server = gethostbyname(argv[1]);
    if (server == NULL) {
        fprintf(stderr,"ERROR, no such host\n");
        exit(1);
    }

    memset(&serv_addr, 0, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    memcpy(&serv_addr.sin_addr.s_addr, server->h_addr, server->h_length);
    serv_addr.sin_port = htons(portno);

    FILE *fp = fopen(argv[3], "rb");
    if (!fp) error("ERROR opening input file");

    struct timeval start;
    gettimeofday(&start, NULL);
    long ts[2] = { start.tv_sec, start.tv_usec };

    // send timestamp
    n = sendto(sockfd, ts, sizeof(ts), 0, (struct sockaddr *)&serv_addr, sizeof(serv_addr));
    if (n < 0) error("ERROR sending timestamp");

    long long total_bytes = 0;
    int seq_num = 0;
    char buffer[BUF_SIZE];

    while ((n = fread(buffer, 1, BUF_SIZE, fp)) > 0) {
        Packet pkt;
        pkt.seq_num = seq_num++;
        pkt.is_eof = 0;
        pkt.length = n;
        memcpy(pkt.data, buffer, n);


        if (sendto(sockfd, &pkt, sizeof(pkt), 0,
               (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0)
            error("ERROR sending packet");
        total_bytes += n;

        char nack_buf[64];
        socklen_t servlen = sizeof(serv_addr);
        int nack_len = recvfrom(sockfd, nack_buf, sizeof(nack_buf), MSG_DONTWAIT,
                        (struct sockaddr *)&serv_addr, &servlen);

        if (nack_len > 0) {
            int missing_seq;
            if (sscanf(nack_buf, "NACK %d", &missing_seq) == 1) {
                printf("Received NACK for seq=%d, resending...\n", missing_seq);

                fseek(fp, (long long)missing_seq * BUF_SIZE, SEEK_SET);
                int reread = fread(buffer, 1, BUF_SIZE, fp);
                Packet resend;
                resend.seq_num = missing_seq;
                resend.is_eof = 0;
                resend.length = reread;
                memcpy(resend.data, buffer, reread);
                sendto(sockfd, &resend, sizeof(resend), 0,
                       (struct sockaddr *)&serv_addr, sizeof(serv_addr));
                fseek(fp, (long long)seq_num * BUF_SIZE, SEEK_SET); // restore position
            }
        }
    }

    printf("Sent %lld bytes via UDP\n", total_bytes);


    // Send EOF marker safely
    Packet eof_pkt;
    memset(&eof_pkt, 0, sizeof(eof_pkt));
    eof_pkt.seq_num = seq_num;
    eof_pkt.is_eof = 1;
    eof_pkt.length = 0;

    for (int i = 0; i < 3; i++) {
        sendto(sockfd, &eof_pkt, HEADER_SIZE, 0,
            (struct sockaddr *)&serv_addr, sizeof(serv_addr));
        usleep(10000); // 10 ms between sends
    }
    printf("EOF marker sent (seq=%d)\n", seq_num);


    fclose(fp);
    close(sockfd);
    return 0;
}
