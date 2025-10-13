/* A simple server in the internet domain using TCP
   The port number is passed as an argument */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h> 
#include <sys/socket.h>
#include <netinet/in.h>

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

    int sockfd, newsockfd, portno;
    socklen_t clilen;
    char buffer[4096];
    struct sockaddr_in serv_addr, cli_addr;
    int n;

    sockfd = socket(AF_INET, SOCK_STREAM, 0);
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

    listen(sockfd, 1);
    clilen = sizeof(cli_addr);
    newsockfd = accept(sockfd,
        (struct sockaddr*)&cli_addr,
        &clilen);
    if (newsockfd < 0)
        error("ERROR on accept");

    FILE *fp = fopen(argv[2], "wb");
    if (fp == NULL) error("ERROR opening output file");

    while ((n = read(newsockfd, buffer, sizeof(buffer))) > 0) {
        fwrite(buffer, 1, n, fp);
    }
    
    prinf("File received successfully.\n");
    fclose(fp);
    close(newsockfd);
    close(sockfd);
    return 0;
}

void* handle_client(void* arg) {
    int sock = *(int*) arg;
    free(arg);

    char buffer[65536];
    int n;
    FILE *fp = fopen("received.bin", "wb");
    if (!fp) {
        perror("File open error");
        close(sock);
        return NULL;
    }

    while ((n = read(sock, buffer, sizeof(buffer))) > 0) {
        fwrite(buffer, 1, n, fp);
    }

    printf("File received from client.\n");
    fclose(fp);
    close(sock);
    return NULL;
}