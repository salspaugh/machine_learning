data <- read.table("lms.dat", header=FALSE)
X <- as.matrix(cbind(data$V1, data$V2))
Y <- as.matrix(data$V3)
Xt.X <- t(X) %*% X
Xt.Y <- t(X) %*% Y
Xt.X.inv <- solve(Xt.X)
theta.star <- Xt.X.inv %*% Xt.Y
print(theta.star)
            [,1]
[1,]  1.0394654
[2,] -0.9764485

data <- read.table("lms.dat", header=FALSE)

X <- as.matrix(cbind(data$V1, data$V2))
Y <- as.matrix(data$V3)

% compute optimal theta
Xt.X <- t(X) %*% X
Xt.X.inv <- solve(t(X) %*% X)
Xt.Y <- t(X) %*% Y
theta.star <- Xt.X.inv %*% Xt.Y

print(theta.star)
            [,1]
[1,]  1.0394654
[2,] -0.9764485

% covariance matrix of X
cov.X <- cov(X)
eigen.cov.X <- eigen(cov.X)

print(eigen.cov.X)
$values
[1] 2.624552 1.026767
$vectors
            [,1]       [,2]
[1,] -0.8530643 -0.5218059
[2,]  0.5218059 -0.8530

% parameter space coordinates
t.x <- seq(theta.star[1]-.5, theta.star[1]+.5, length=100)
t.y <- seq(theta.star[2]-.5, theta.star[2]+.5, length=100)

% the loss at point (i, j) in parameter space
loss.at <- function(i, j) {
    theta.hat <- rbind(t.x[i], t.y[j])
    Y.hat <- X %*% theta.hat
    Y.err <- Y - Y.hat
    loss <- t(Y.err) %*% Y.err
    return(loss)
}

% computes loss over all parameter space coordinates from above
compute.loss <- function() {
    Z <- matrix(nrow=100, ncol=100)
    for (i in seq(1,100)) {
        for(j in seq(1,100)) {
            Z[i,j] <- loss.at(i,j)
        }
    }
    return(Z)
}

Z <- compute.loss()

pdf("ps2_1b.pdf")
contour(t.x, t.y, Z)
dev.off()

data <- read.table("lms.dat", header=FALSE)
X <- as.matrix(cbind(data$V1, data$V2))
Y <- as.matrix(data$V3)

% compute optimal theta
Xt.X <- t(X) %*% X
Xt.X.inv <- solve(t(X) %*% X)
Xt.Y <- t(X) %*% Y
theta.star <- Xt.X.inv %*% Xt.Y

% compute step sizes to use
cov.X <- cov(X)
eigen.cov.X <- eigen(cov.X)
lambda.max <- max(eigen.cov.X$values)
rho.max <- 1/lambda.max
rho.half <- rho.max/2
rho.quart <- rho.max/4

% returns a step in the right direction
step <- function(rho, theta.t) {
     i <- sample(1:dim(X)[1], 1)
     x.i <- as.matrix(X[i,])
     y.i <- as.matrix(Y[i,])
     y.err <- y.i - (t(theta.t) %*% x.i)
     s <- rho * (y.err[1,1] * x.i)
     return(s)
}

% take iters number of LMS steps
build.lms.path <- function(rho, iters) {
    theta.path.1 <- vector(mode="numeric", length=iters)
    theta.path.2 <- vector(mode="numeric", length=iters)
    theta.curr <- as.matrix(c(theta.path.1[1], theta.path.2[1]))
    for (i in 2:iters) {
        s <- step(rho, theta.curr)
        theta.next <- theta.curr + s
        theta.path.1[i] <- theta.next[1]
        theta.path.2[i] <- theta.next[2]
        theta.curr <- theta.next
    }
    return(rbind(theta.path.1, theta.path.2))
}

% plot the path
plot.lms.path <- function(out, iters, stepsize) {
    lms <- build.lms.path(stepsize, iters)
    pdf(out)
    plot(lms[1,], lms[2,], typ="l")
    points(theta.star[1], theta.star[2], pch=1, cex=3, col="red", lwd=4)
    dev.off()
}

plot.lms.path("ps2_1c_rho_max.pdf", 100, rho.max) 
plot.lms.path("ps2_1c_rho_half.pdf", 100, rho.half) 
plot.lms.path("ps2_1c_rho_quart.pdf", 100, rho.quart) 

savehistory("ps2_1c.R")
