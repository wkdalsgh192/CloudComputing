/* A simple server in the internet domain using TCP
   The port number is passed as an argument */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h> 
#include <sys/socket.h>
#include <sys/time.h>
#include <netinet/in.h>

#define BUF_SIZE 65536

void error(const char* msg)
{
    perror(msg);
    exit(1);
}

int main(int argc, char* argv[])
{
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <port> <output_file>\n", argv[0]);
        exit(1);
    }

    int sockfd, newsockfd, portno, n;
    socklen_t clilen;
    char buffer[BUF_SIZE];
    struct sockaddr_in serv_addr, cli_addr;

    sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0)
        error("ERROR opening socket");

    memset(&serv_addr, 0, sizeof(serv_addr));
    portno = atoi(argv[1]);
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = INADDR_ANY;
    serv_addr.sin_port = htons(portno);
    if (bind(sockfd, (struct sockaddr*)&serv_addr,
        sizeof(serv_addr)) < 0)
        error("ERROR on binding");

    printf("UDP server listening on port %d...\n", portno);

    FILE *fp = fopen(argv[2], "wb");
    if (fp == NULL) error("ERROR opening output file");

    long ts[2];
    clilen = sizeof(cli_addr);
    n = recvfrom(sockfd, ts, sizeof(ts), 0, (struct sockaddr*)&cli_addr, &clilen);
    if (n != sizeof(ts)) error("ERROR receiving timestamp");

    // read the client’s start timestamp
    struct timeval start, end;
    start.tv_sec = ts[0];
    start.tv_usec = ts[1];
    printf("Server got start: %ld.%06ld\n", start.tv_sec, start.tv_usec);
    

    long long total_bytes = 0;
    while ((n = recvfrom(newsockfd, buffer, sizeof(buffer), 0, (struct sockaddr*)&cli_addr, &clilen)) > 0) {
        fwrite(buffer, 1, n, fp);
        total_bytes += n;

        if (n < BUF_SIZE) break;
    }
    
    gettimeofday(&end, NULL);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec)/1e6;


    printf("Received %lld bytes in %.2f seconds\n", total_bytes, elapsed);
    double mbps = (total_bytes * 8.0) / (elapsed * 1e6);
    printf("Throughput: %.2f Mbps\n", mbps);

    fclose(fp);
    close(newsockfd);
    close(sockfd);
    return 0;
}