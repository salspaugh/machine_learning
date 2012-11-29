cdata <- read.table("classification2d.dat", header=FALSE)
names(cdata) <- c("X1", "X2", "class")

% for plotting x's and o's
ones <- subset(cdata, class == 1)
zeros <- subset(cdata, class == 0)

X <- as.matrix(cbind(cdata$X1, cdata$X2))
cov.X <- cov(X)
eigen.cov.X <- eigen(cov.X)
max.lambda <- max(eigen.cov.X$values)
rho <- 1/max.lambda
rho <- rho/4

% the logistic step
step <- function(theta.t) {
     i <- sample(1:dim(X)[1], 1)
     x.i <- as.matrix(X[i,])
     y.i <- as.matrix(Y[i,])
     theta.T.x <- t(theta.t) %*% x.i
     y.err <- y.i - (1 / (1 + exp(-1*theta.T.x)))
     s <- rho * (y.err[1,1] * x.i)
     return(s)
}

% LMS iters times
build.lms.path <- function(iters) {
    theta.path.1 <- vector(mode="numeric", length=iters)
    theta.path.2 <- vector(mode="numeric", length=iters)
    theta.curr <- as.matrix(c(theta.path.1[1], theta.path.2[1]))
    for (i in 2:iters) {
        s <- step(theta.curr)
        theta.next <- theta.curr + s
        theta.path.1[i] <- theta.next[1]
        theta.path.2[i] <- theta.next[2]
        theta.curr <- theta.next
    }
    return(rbind(theta.path.1, theta.path.2))
}

lms.path <- build.lms.path(100)

% for plotting the line = .5
theta.bar <- lms.path[,100]
slope <- -1* theta.bar[1] / theta.bar[2]
intercept <- .5 / theta.bar[2]

% for solving the linear regression and plotting the line = .5
Y <- as.matrix(cdata$class)
Xt.X <- t(X) %*% X
Xt.Y <- t(X) %*% Y
Xt.X.inv <- solve(Xt.X)
theta.star <- Xt.X.inv %*% Xt.Y
slope.lin <- -1 * theta.star[1,1] / theta.star[2,1]
intercept.lin <- .5 / theta.star[2,1]

% plot the points and both lines
plot.classification <- function() {
    pdf("ps2_2.pdf")
    plot(ones$X1, ones$X2, typ="p", pch="x")
    points(zeros$X1, zeros$X2, pch="o")
    abline(intercept, slope)
    abline(intercept.lin, slope.lin, lty=2)
    dev.off()
}

% try it on the new data
tdata <- read.table("testing.dat", header=FALSE)
names(tdata) <- c("X1", "X2", "class")
new.zeros <- subset(tdata, class == 0)
new.ones <- subset(tdata, class == 1)
plot.new.classification <- function() {
   pdf("ps2_2d.pdf")
   plot(new.ones$X1, new.ones$X2, typ="p", pch="x", xlim=c(-4,4), ylim=c(-4, 4))
   points(new.zeros$X1, new.zeros$X2, pch="o")
   abline(intercept, slope)
   abline(intercept.lin, slope.lin, lty=2)
   dev.off()
}

plot.new.classification()
