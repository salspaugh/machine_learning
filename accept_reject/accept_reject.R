# (b) Implement an accept-reject sampler as specified.  

sampler <- function() {
    ret <- list()
    draws <- 0 
    sampled <- FALSE
    while (!sampled) {
       u <- runif(n=1)
       # The distribution we are sampling from is 6x(1-x).
       # Setting M = 3 means when u = 1/2 we have a 1/2 chance of accepting.
       x <- (6*u*(1-u)) / 3
       v <- runif(n=1)
       draws <- draws + 2
       if (v < x) { sampled <- TRUE }
   }
   ret[[1]] <- u # the sample
   ret[[2]] <- draws # amount uniform sampler was called
   return(ret)
}

n <- 10000 # number of samples
samples <- vector(mode="numeric", length=n)
actual.draws <- 0 # number of times uniform sampler is called

# Do the sampling.
for (i in seq(1,n)) {
   r <- sampler()
   samples[i] <- r[[1]]
   actual.draws <- actual.draws + r[[2]]
}

# Plot the sample histogram.
pdf("q3.pdf")
hist(samples)
dev.off()

# (c) Count the actual number of times per sample the uniform sampler was called.
# > actual.draws / n
# [1] 5.9328
