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
    if (argc < 4) {
        fprintf(stderr,"Usage: %s <port> <output_file> <interface>\n", argv[0]);
        exit(1);
    }

    int sockfd, portno, n;
    socklen_t clilen;
    struct sockaddr_in serv_addr, cli_addr;

    // get MTU and calculate safe payload
    int mtu = get_mtu(argv[3]);
    int BUF_SIZE = mtu - 28;
    if (BUF_SIZE > 9000) BUF_SIZE = 9000;
    char *buffer = malloc(BUF_SIZE);

    printf("Detected MTU = %d, using UDP chunk size = %d bytes\n", mtu, BUF_SIZE);

    portno = atoi(argv[1]);
    sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0) error("ERROR opening socket");

    memset(&serv_addr, 0, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = INADDR_ANY;
    serv_addr.sin_port = htons(portno);

    if (bind(sockfd, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) < 0)
        error("ERROR on binding");

    printf("UDP server listening on port %d...\n", portno);

    FILE *fp = fopen(argv[2], "wb");
    if (!fp) error("ERROR opening output file");

    long ts[2];
    clilen = sizeof(cli_addr);
    n = recvfrom(sockfd, ts, sizeof(ts), 0, (struct sockaddr*)&cli_addr, &clilen);
    if (n != sizeof(ts)) error("ERROR receiving timestamp");

    struct timeval start, end;
    start.tv_sec  = ts[0];
    start.tv_usec = ts[1];
    printf("Server got client start: %ld.%06ld\n", start.tv_sec, start.tv_usec);

    long long total_bytes = 0;
    while ((n = recvfrom(sockfd, buffer, BUF_SIZE, 0, (struct sockaddr*)&cli_addr, &clilen)) > 0) {
        
        if (n == 4 && strncmp(buffer, "EOF", 3) == 0) {
            printf("End of file marker received.\n");
            break;
        }
        
        fwrite(buffer, 1, n, fp);
        total_bytes += n;

        if (n < BUF_SIZE) break; // likely last chunk
    }

    gettimeofday(&end, NULL);
    double elapsed = (end.tv_sec - start.tv_sec) +
                     (end.tv_usec - start.tv_usec)/1e6;

    printf("Received %lld bytes in %.2f seconds\n", total_bytes, elapsed);
    double mbps = (total_bytes * 8.0) / (elapsed * 1e6);
    printf("Throughput: %.2f Mbps\n", mbps);

    free(buffer);
    fclose(fp);
    close(sockfd);
    return 0;
}
