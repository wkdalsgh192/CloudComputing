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
    char *buffer = malloc(BUF_SIZE);

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
    while ((n = fread(buffer, 1, BUF_SIZE, fp)) > 0) {
        if (sendto(sockfd, buffer, n, 0, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0)
            error("ERROR sending file data");
        total_bytes += n;
    }

    printf("Sent %lld bytes via UDP\n", total_bytes);

    char eof_marker[] = "EOF";
    sendto(sockfd, eof_marker, sizeof(eof_marker), 0, (struct sockaddr *)&serv_addr, sizeof(serv_addr));

    free(buffer);
    fclose(fp);
    close(sockfd);
    return 0;
}
