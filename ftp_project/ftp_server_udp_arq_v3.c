// Minimal reliable UDP server (Stop-and-Wait ARQ)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <net/if.h>
#include <sys/time.h>

#define MAX_CHUNK 9000   // safety cap
#define ACK_DATA  1
#define ACK_EOF   2
#define MSG_NACK  3

typedef struct {
    uint32_t seq;     // sequence number of this packet
    uint32_t is_eof;  // 0=data, 1=eof
    uint32_t len;     // bytes valid in data[]
    unsigned char data[MAX_CHUNK];
} __attribute__((packed)) UdpPkt;

typedef struct {
    uint32_t kind;    // ACK_DATA, ACK_EOF, or MSG_NACK
    uint32_t next;    // next expected seq from server's POV
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
    if (argc < 4){
        fprintf(stderr,"Usage: %s <port> <output_file> <interface>\n", argv[0]);
        return 1;
    }

    // compute safe chunk size
    int mtu = get_mtu(argv[3]);
    int CHUNK = mtu - 28;           // IP(20)+UDP(8)
    if (CHUNK > MAX_CHUNK) CHUNK = MAX_CHUNK;
    printf("Server: MTU=%d, chunk=%d\n", mtu, CHUNK);

    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) die("socket");

    // small receive timeout to allow clean exit if client dies
    struct timeval rcvto = { .tv_sec=0, .tv_usec=300000 }; // 300 ms
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &rcvto, sizeof(rcvto));

    struct sockaddr_in me = {0}, cli = {0};
    me.sin_family = AF_INET;
    me.sin_addr.s_addr = INADDR_ANY;
    me.sin_port = htons((uint16_t)atoi(argv[1]));
    if (bind(sock, (struct sockaddr*)&me, sizeof(me)) < 0) die("bind");

    FILE *fp = fopen(argv[2], "wb");
    if (!fp) die("fopen");

    socklen_t clen = sizeof(cli);
    uint32_t expect = 0;
    long long total = 0;

    while (1){
        UdpPkt pkt;
        ssize_t n = recvfrom(sock, &pkt, sizeof(pkt), 0, (struct sockaddr*)&cli, &clen);
        if (n < 0) continue; // timeout -> keep waiting

        // data packet or EOF
        if (pkt.is_eof){
            // ACK EOF only after all in-order data received
            AckPkt ack = { .kind = ACK_EOF, .next = expect };
            sendto(sock, &ack, sizeof(ack), 0, (struct sockaddr*)&cli, clen);
            if (pkt.seq == expect){  // client’s EOF seq equals next expected -> done
                printf("Server: EOF for seq=%u, done.\n", pkt.seq);
                break;
            } else {
                // We’re still missing data; client should keep sending/resending
                printf("Server: EOF seen but expect=%u (missing data).\n", expect);
                continue;
            }
        }

        // normal data
        if (pkt.seq == expect){
            // in-order -> write and advance, ACK next expected
            size_t w = fwrite(pkt.data, 1, pkt.len, fp);
            (void)w;
            total += pkt.len;
            expect++;
            AckPkt ack = { .kind = ACK_DATA, .next = expect };
            sendto(sock, &ack, sizeof(ack), 0, (struct sockaddr*)&cli, clen);
        } else if (pkt.seq < expect){
            // duplicate -> re-ACK current next to stop extra retransmissions
            AckPkt ack = { .kind = ACK_DATA, .next = expect };
            sendto(sock, &ack, sizeof(ack), 0, (struct sockaddr*)&cli, clen);
        } else {
            // future packet -> ask for missing one
            AckPkt nack = { .kind = MSG_NACK, .next = expect };
            sendto(sock, &nack, sizeof(nack), 0, (struct sockaddr*)&cli, clen);
        }
    }

    fclose(fp);
    close(sock);
    printf("Server: received %lld bytes\n", total);
    return 0;
}
