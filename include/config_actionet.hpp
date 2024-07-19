#ifndef CONFIG_ACTIONET_HPP
#define CONFIG_ACTIONET_HPP

#define stdout_printf printf
#define stderr_printf printf
#define FLUSH fflush(stdout)

#define SYS_THREADS_DEF (std::thread::hardware_concurrency() - 2)

// #define stdout_printf Rprintf
// #define stderr_printf REprintf
// #define FLUSH R_FlushConsole()

#endif
