#include <algorithm>
#include <iostream>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <netinet/in.h>
#include <netdb.h>
#include <vector>

void sigchld_handler(int s) {
    int saved_errno = errno;
    while (waitpid(-1, NULL, WNOHANG) > 0);
    errno = saved_errno;
}

/**
 * Get sockaddr, IPv4 or IPv6
 */
void *get_in_addr(struct sockaddr *sa) {
    if (sa->sa_family == AF_INET) {
        return &(((struct sockaddr_in*)sa)->sin_addr);
    }
    return &(((struct sockaddr_in6*)sa)->sin6_addr);
}

/**
 * Get the hints struct to pass to getaddrinfo()
 */
struct addrinfo get_hints(bool stream_socket = true) {
    struct addrinfo hints = {};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = stream_socket ? SOCK_STREAM : SOCK_DGRAM;
    hints.ai_flags = AI_PASSIVE;
    return hints;
}

/**
 * Given an addrinfo struct obtained from getaddrinfo(), bind to the
 * first socket available
 */
int bind_first(struct addrinfo *serv_info, bool stream_socket = true) {
    struct addrinfo *p;
    int sock_fd = -1;
    int yes = 1;

    for (p = serv_info; p != nullptr; p = p->ai_next) {
        if ((sock_fd = socket(p->ai_family, p->ai_socktype, p->ai_protocol)) == -1) {
            perror("Server: socket() failed");
            continue;
        }

        if (stream_socket && setsockopt(sock_fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(int)) == -1) {
            perror("Server: setsockopt() failed");
            exit(1);
        }

        if (bind(sock_fd, p->ai_addr, p->ai_addrlen) == -1) {
            close(sock_fd);
            perror("Server: bind() failed");
            continue;
        }
        break;
    }
    if (p == nullptr) {
        std::cout << "Server: bind() failed" << std::endl;
        exit(1);
    }

    freeaddrinfo(serv_info);
    return sock_fd;
}

/**
 * Connect to the first valid socket() call
 */
int connect_first(struct addrinfo *serv_info) {
    struct addrinfo *p;
    int sock_fd = -1;
    for (p = serv_info; p != nullptr; p = p->ai_next) {
        if ((sock_fd = socket(p->ai_family, p->ai_socktype, p->ai_protocol)) == -1) {
            perror("Client: socket() failed");
            continue;
        }

        if (connect(sock_fd, p->ai_addr, p->ai_addrlen) == -1) {
            close(sock_fd);
            perror("Client: connect() failed");
            continue;
        }
        break;
    }
    if (p == nullptr) {
        std::cout << "Client: connect() failed" << std::endl;
    }
    return sock_fd;
}