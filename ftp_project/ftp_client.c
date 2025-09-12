#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <netinet/in.h>
#include <netdb.h> 

#define BUF_SIZE 65536

void error(const char* msg)
{
    perror(msg);
    exit(0);
}

int main(int argc, char* argv[])
{

    if (argc < 4) {
        fprintf(stderr, "Usage: %s <hostname> <port> <file_to_send>\n", argv[0]);
        exit(0);
    }

    int sockfd, portno, n;
    struct sockaddr_in serv_addr;
    struct hostent* server;

    char buffer[BUF_SIZE];

    portno = atoi(argv[2]);
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0)
        error("ERROR opening socket");
    server = gethostbyname(argv[1]);
    if (server == NULL) {
        fprintf(stderr, "ERROR, no such host\n");
        exit(0);
    }

    memset(&serv_addr, 0, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    memcpy(&serv_addr.sin_addr.s_addr, server->h_addr_list[0], server->h_length);
    serv_addr.sin_port = htons(portno);

    if (connect(sockfd, (struct sockaddr *) &serv_addr, sizeof(serv_addr)) < 0)
        error("ERROR connecting");

    FILE *fp = fopen(argv[3], "rb");
    if (fp == NULL) error("ERROR opening input file");

    struct timeval start;
    gettimeofday(&start, NULL);

    long sec = start.tv_sec;
    long usec = start.tv_usec;
    printf("Client start: %ld.%06ld\n", sec, usec);
    write(sockfd, &sec, sizeof(sec));
    write(sockfd, &usec, sizeof(usec));

    if (write(sockfd, &start, sizeof(start) != sizeof(start))) {
        error("ERROR sending timestamp to server");
    };

    long long total_bytes = 0;
    while ((n=fread(buffer, 1, sizeof(buffer), fp)) > 0) {
        if (write(sockfd, buffer, n) < 0) error("ERROR writing to socket");
        total_bytes += n;
    }

    printf("File sent successfully. \n");
    fclose(fp);
    close(sockfd);
    return 0;
}