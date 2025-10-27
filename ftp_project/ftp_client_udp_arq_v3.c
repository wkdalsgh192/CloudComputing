// Minimal reliable UDP client (Stop-and-Wait ARQ)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <net/if.h>
#include <sys/time.h>

#define MAX_CHUNK 9000
#define ACK_DATA  1
#define ACK_EOF   2
#define MSG_NACK  3

typedef struct {
    uint32_t seq;
    uint32_t is_eof;
    uint32_t len;
    unsigned char data[MAX_CHUNK];
} __attribute__((packed)) UdpPkt;

typedef struct {
    uint32_t kind;
    uint32_t next;
} __attribute__((packed)) AckPkt;

static void die(const char *s){ perror(s); exit(1); }

static int get_mtu(const char *ifname){
    int s = socket(AF_INET, SOCK_DGRAM, 0);
    if (s<0) die("socket");
    struct ifreq ifr; memset(&ifr,0,sizeof(ifr));
    strncpy(ifr.ifr_name, ifname, IFNAMSIZ-1);
    if (ioctl(s, SIOCGIFMTU, &ifr) < 0){ close(s); die("ioctl SIOCGIFMTU"); }
    close(s);
    return ifr.ifr_mtu;
}

int main(int argc, char **argv){
    if (argc < 5){
        fprintf(stderr,"Usage: %s <server_ip> <port> <file> <interface>\n", argv[0]);
        return 1;
    }

    int mtu = get_mtu(argv[4]);
    int CHUNK = mtu - 28;
    if (CHUNK > MAX_CHUNK) CHUNK = MAX_CHUNK;
    printf("Client: MTU=%d, chunk=%d\n", mtu, CHUNK);

    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) die("socket");

    // bind to interface (nice-to-have; ignore failure)
    struct ifreq ifr; memset(&ifr,0,sizeof(ifr));
    strncpy(ifr.ifr_name, argv[4], IFNAMSIZ-1);
    setsockopt(sock, SOL_SOCKET, SO_BINDTODEVICE, &ifr, sizeof(ifr));

    // small receive timeout for ACKs
    struct timeval rcvto = { .tv_sec=0, .tv_usec=300000 }; // 300 ms
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &rcvto, sizeof(rcvto));

    struct sockaddr_in srv = {0};
    srv.sin_family = AF_INET;
    srv.sin_port = htons((uint16_t)atoi(argv[2]));
    if (inet_pton(AF_INET, argv[1], &srv.sin_addr) != 1) die("inet_pton");
    socklen_t slen = sizeof(srv);

    FILE *fp = fopen(argv[3], "rb");
    if (!fp) die("fopen");

    uint32_t seq = 0;
    long long total = 0;

    // send chunks one-by-one, wait ACK each time
    while (1){
        UdpPkt pkt; memset(&pkt, 0, sizeof(pkt));
        pkt.seq = seq;
        pkt.len = fread(pkt.data, 1, CHUNK, fp);
        if (pkt.len == 0){  // EOF
            pkt.is_eof = 1;
            ssize_t sent = sendto(sock, &pkt, sizeof(uint32_t)*3, 0,
                                  (struct sockaddr*)&srv, slen);
            (void)sent;

            // wait for ACK_EOF, retransmit on timeout/NACK
            for (;;){
                AckPkt ack;
                ssize_t n = recvfrom(sock, &ack, sizeof(ack), 0,
                                     (struct sockaddr*)&srv, &slen);
                if (n > 0 && ack.kind == ACK_EOF && ack.next == seq){
                    printf("Client: EOF ACKed.\n");
                    goto done;
                }
                // resend EOF
                sendto(sock, &pkt, sizeof(uint32_t)*3, 0, (struct sockaddr*)&srv, slen);
            }
        }

        // send data packet (header + payload)
        ssize_t sent = sendto(sock, &pkt, sizeof(uint32_t)*3 + pkt.len, 0,
                              (struct sockaddr*)&srv, slen);
        (void)sent;

        // wait for ACK_DATA for next==seq+1, else retransmit
        for (;;){
            AckPkt ack;
            ssize_t n = recvfrom(sock, &ack, sizeof(ack), 0,
                                 (struct sockaddr*)&srv, &slen);
            if (n > 0){
                if ((ack.kind == ACK_DATA && ack.next == seq+1) ||
                    (ack.kind == MSG_NACK && ack.next == seq)){
                    break; // OK to advance or resend; fall through to logic below
                }
            }
            // timeout or wrong ack -> retransmit current packet
            sendto(sock, &pkt, sizeof(uint32_t)*3 + pkt.len, 0,
                   (struct sockaddr*)&srv, slen);
        }

        total += pkt.len;
        seq++;
    }

done:
    fclose(fp);
    close(sock);
    printf("Client: sent %lld bytes\n", total);
    return 0;
}
