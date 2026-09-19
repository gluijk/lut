# Building, applying and visualizing image processing LUTs
# www.overfitting.net
# https://www.overfitting.net/2026/09/ingenieria-inversa-de-procesado-de.html


library(Rcpp)
library(png)  # standard HaldCLUT format (only writes 8-bit PNG)
library(tiff)  # to write 16-bit files (both HaldCLUT and images)
library(rgl)  # for LUT visualization


# C++ acceleration functions
Sys.setenv("PKG_CXXFLAGS" = "-fopenmp")
Sys.setenv("PKG_LIBS" = "-fopenmp")

sourceCpp("build_LUT.cpp")
sourceCpp("apply_LUT.cpp")


# 1. Generate a LUT from a pair of images (input/output) both in .cube and HaldCLUT formats
build_lut <- function(img_in_path, 
                     img_out_path, 
                     output_base_path, 
                     N = 33, 
                     k_neighbors = 8, 
                     power = 2.0, 
                     apply_smooth = FALSE, sigma = 0.8,  # gaussian blur parameter
                     force_0 = FALSE, force_1 = FALSE) {  # force (0,0,0) and/or (1,1,1)
    
    cat("Cargando imágenes...\n")
    img1 <- readTIFF(img_in_path)
    img2 <- readTIFF(img_out_path)
    
    mat_in  <- matrix(img1, ncol = 3)
    mat_out <- matrix(img2, ncol = 3)
    
    cat("Generando LUT 3D (N =", N, ") e interpolando mediante IDW + Suavizado Gaussiano...\n")
    lut_data <- build_lut3d_cpp(mat_in, mat_out, N, k_neighbors, power, apply_smooth, sigma, force_0, force_1)
    
    # --- 1. Exportar en formato .cube ---
    cube_path <- paste0(output_base_path, ".cube")
    con <- file(cube_path, "w")
    writeLines(c(
        '# Created with Rcpp, IDW and Gaussian 3D Blur',
        paste0('TITLE "Generated_LUT_', N, 'x', N, 'x', N, '"'),
        paste0('LUT_3D_SIZE ', N),
        ''
    ), con)
    
    lut_matrix <- matrix(lut_data, ncol = 3, byrow = TRUE)
    lines <- sprintf("%.6f %.6f %.6f", lut_matrix[, 1], lut_matrix[, 2], lut_matrix[, 3])
    writeLines(lines, con)
    close(con)
    cat("✓ Archivo .cube guardado en:", cube_path, "\n")
    
    # --- 2. Exportar en formato HaldCLUT (PNG) ---
    png_path <- paste0(output_base_path, "_hald.png")
    tif_path <- paste0(output_base_path, "_hald.tif")
    
    # HaldCLUT requiere un lado de imagen = N^1.5 en píxeles (ej: N=64 -> 512x512)
    side <- round(N^(1.5))
    hald_img <- array(0, dim = c(side, side, 3))
    
    idx <- 1
    for (y in 1:side) {
        for (x in 1:side) {
            if (idx <= nrow(lut_matrix)) {
                hald_img[y, x, 1] <- lut_matrix[idx, 1]
                hald_img[y, x, 2] <- lut_matrix[idx, 2]
                hald_img[y, x, 3] <- lut_matrix[idx, 3]
                idx <- idx + 1
            }
        }
    }
    
    writePNG(hald_img, png_path)  # unfortunately only 8-bit PNGs are possible with writePNG()
    writeTIFF(hald_img, tif_path, bits.per.sample = 16)  # save as 16-bit PNG later
    cat("✓ Archivo HaldCLUT PNG (8 bits) guardado en:", png_path, "\n")
    cat("✓ Archivo HaldCLUT TIF (16 bits) guardado en:", tif_path, "\n")
}


# 2. Apply a LUT to an image either in .cube or HaldCLUT format
apply_lut <- function(input_image_path, lut_path, output_image_path) {
    
    # 1. Read Input Image
    ext_img <- tolower(tools::file_ext(input_image_path))
    if (ext_img %in% c("tif", "tiff")) {
        img <- readTIFF(input_image_path)
    } else if (ext_img == "png") {
        img <- readPNG(input_image_path)[,,1:3]  # drop alpha channel if present
    } else {
        stop("Unsupported input format. Use TIFF or PNG.")
    }
    
    img_dims <- dim(img)
    img_mat  <- matrix(img, ncol = 3)
    
    # 2. Process according to LUT format
    ext_lut <- tolower(tools::file_ext(lut_path))
    
    if (ext_lut == "cube") {
        cat("Applying .cube LUT via trilinear interpolation...\n")
        out_mat <- apply_lut_cube_cpp(img_mat, lut_path)
    } else if (ext_lut == "png") {
        cat("Reading HaldCLUT PNG...\n")
        hald_img <- readPNG(lut_path)
        if (length(dim(hald_img)) == 3 && dim(hald_img)[3] >= 3) {
            hald_img <- hald_img[,,1:3]
        }
        
        hald_dim <- dim(hald_img)[1]
        
        # HaldCLUT dimension side = N^1.5
        N <- round(hald_dim^(2/3))
        
        # CORRECCIÓN: Reorganizar los ejes a [Canales, Ancho, Alto] 
        # antes de aplanar a vector 1D para C++
        lut_vector <- as.vector(aperm(hald_img, c(3, 2, 1)))
        
        cat("Applying HaldCLUT (N =", N, ") via trilinear interpolation...\n")
        out_mat <- apply_lut_array_cpp(img_mat, lut_vector, N)
        
    } else {
        stop("Unsupported LUT extension. Use .cube or .png (HaldCLUT).")
    }
    
    # 3. Reconstruct image array and save to disk
    out_array <- array(out_mat, dim = img_dims)
    
    ext_out <- tolower(tools::file_ext(output_image_path))
    if (ext_out %in% c("tif", "tiff")) {
        writeTIFF(out_array, output_image_path, bits.per.sample = 16)
    } else if (ext_out == "png") {
        writePNG(out_array, output_image_path)
    } else {
        stop("Unsupported output format. Use TIFF or PNG.")
    }
    
    cat("✓ Transformed image saved to:", output_image_path, "\n")
}


# 3. LUT 3D Visualization from .cube format LUT

# Para una LUT `33³` empezaría con grid = 7, es suficientemente densa para revelar
# la geometría de la transformación sin convertir la escena en una maraña de 35.937 elementos.

# ============================================================
# READ .CUBE 3D LUT
# ============================================================
read_cube <- function(file) {
    x <- readLines(file, warn = FALSE)
    # Remove leading/trailing spaces
    x <- trimws(x)
    # Remove empty lines and comments
    x <- x[
        x != "" &
            !grepl("^#", x)
    ]
    
    # ----------------------------------------------------------
    # LUT size
    # ----------------------------------------------------------
    z <- grep(
        "^LUT_3D_SIZE\\s+",
        x,
        value = TRUE
    )
    
    if (length(z) != 1) stop("LUT_3D_SIZE not found.")
    
    N <- as.integer(
        sub(
            "^LUT_3D_SIZE\\s+",
            "",
            z
        )
    )
    
    if (is.na(N) || N < 2) stop("Invalid LUT_3D_SIZE.")
    
    # ----------------------------------------------------------
    # Numerical LUT data
    # ----------------------------------------------------------
    numeric <- grepl(
        "^[-+]?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?[[:space:]]+",
        x
    )
    
    d <- x[numeric]
    lut <- do.call(
        rbind,
        lapply(
            strsplit(d, "\\s+"),
            as.numeric
        )
    )
    
    if (nrow(lut) != N^3) {
        stop(
            "Expected ",
            N^3,
            " LUT entries, found ",
            nrow(lut),
            "."
        )
    }
    
    if (ncol(lut) != 3)
        stop(
            "Each LUT entry must contain three values: R G B."
        )
    
    # ----------------------------------------------------------
    # .cube ordering
    #
    # R varies fastest
    # G varies next
    # B varies slowest
    #
    # Therefore:
    # lut[R,G,B,channel]
    # ----------------------------------------------------------
    lut <- array(
        lut,
        dim = c(N, N, N, 3)
    )
    
    invisible(lut)
}

# ============================================================
# PLOT 3D LUT
# ============================================================
plot_lut <- function(
        file,
        grid = 7,
        show_input = TRUE,
        show_output = TRUE,
        show_greys = TRUE,
        show_vertices = TRUE,
        show_vectors = FALSE,
        vector_step = 4,
        line_width = 2,
        input_alpha = 0.30) {
    
    # ----------------------------------------------------------
    # Read LUT
    # ----------------------------------------------------------
    lut <- read_cube(file)
    N <- dim(lut)[1]
    
    # ----------------------------------------------------------
    # Input RGB coordinates
    # ----------------------------------------------------------
    u <- seq(
        0,
        1,
        length.out = N
    )
    
    # ----------------------------------------------------------
    # Grid density
    #
    # For example:
    # grid = 7
    # means 7 points along each dimension.
    # ----------------------------------------------------------
    idx <- unique(
        round(
            seq(
                1,
                N,
                length.out = grid
            )
        )
    )
    
    # ----------------------------------------------------------
    # Open 3D window
    # ----------------------------------------------------------
    open3d()
    bg3d("white")
    
    # ==========================================================
    # ORIGINAL RGB CUBE (8 VERTICES)
    # ==========================================================
    if (show_input) {
        # Cube vertices
        V <- rbind(
            c(0, 0, 0),
            c(1, 0, 0),
            c(1, 1, 0),
            c(0, 1, 0),
            
            c(0, 0, 1),
            c(1, 0, 1),
            c(1, 1, 1),
            c(0, 1, 1)
        )
        
        # Cube edges
        E <- rbind(
            c(1, 2),
            c(2, 3),
            c(3, 4),
            c(4, 1),
            
            c(5, 6),
            c(6, 7),
            c(7, 8),
            c(8, 5),
            
            c(1, 5),
            c(2, 6),
            c(3, 7),
            c(4, 8)
        )
        
        # Draw cube edges
        for (e in seq_len(nrow(E))) {
            segments3d(
                x = V[E[e, ], 1],
                y = V[E[e, ], 2],
                z = V[E[e, ], 3],
                col = adjustcolor(
                    "grey50",
                    alpha.f = input_alpha
                ),
                lwd = 1
            )
        }
        
        # Deactivated because they were a bit messy
        # # Draw cube vertices
        # spheres3d(
        #     V[, 1],
        #     V[, 2],
        #     V[, 3],
        #     radius = 0.012,
        #     color = adjustcolor(
        #         "grey50",
        #         alpha.f = input_alpha
        #     )
        # )
    }
    
    # ==========================================================
    # FUNCTION TO DRAW A TRANSFORMED LINE
    # ==========================================================
    draw_line <- function(points) {
        # Each segment gets the colour of its
        # corresponding output RGB value.
        for (s in seq_len(nrow(points) - 1)) {
            col <- rgb(
                pmax(
                    0,
                    pmin(
                        1,
                        points[s, 1]
                    )
                ),
                
                pmax(
                    0,
                    pmin(
                        1,
                        points[s, 2]
                    )
                ),
                
                pmax(
                    0,
                    pmin(
                        1,
                        points[s, 3]
                    )
                )
            )
            
            segments3d(
                x = points[s:(s + 1), 1],
                y = points[s:(s + 1), 2],
                z = points[s:(s + 1), 3],
                col = col,
                lwd = line_width
            )
        }
    }
    
    # ==========================================================
    # TRANSFORMED RGB GRID
    # ==========================================================
    if (show_output) {
        # --------------------------------------------------------
        # R direction
        #
        # R varies
        # G and B remain constant
        # --------------------------------------------------------
        for (g in idx) {
            for (b in idx) {
                points <- t(
                    sapply(
                        idx,
                        function(r)
                            lut[r, g, b, ]
                    )
                )
                draw_line(points)
            }
        }
        
        # --------------------------------------------------------
        # G direction
        #
        # G varies
        # R and B remain constant
        # --------------------------------------------------------
        for (r in idx) {
            for (b in idx) {
                points <- t(
                    sapply(
                        idx,
                        function(g)
                            lut[r, g, b, ]
                    )
                )
                draw_line(points)
            }
        }
        
        # --------------------------------------------------------
        # B direction
        #
        # B varies
        # R and G remain constant
        # --------------------------------------------------------
        for (r in idx) {
            for (g in idx) {
                points <- t(
                    sapply(
                        idx,
                        function(b)
                            lut[r, g, b, ]
                    )
                )
                draw_line(points)
            }
        }
    }
    
    # ==========================================================
    # GREYSCALE / NEUTRAL AXIS
    #
    # Input:
    # R = G = B
    #
    # Output:
    # LUT(R,R,R)
    # ==========================================================
    if (show_greys) {
        grey_idx <- idx
        
        # --------------------------------------------------------
        # Original neutral axis
        # --------------------------------------------------------
        if (show_input) {
            lines3d(
                x = u[grey_idx],
                y = u[grey_idx],
                z = u[grey_idx],
                col = "grey60",
                lwd = 4,
                alpha = 0.5
            )
        }
        
        # --------------------------------------------------------
        # Transformed neutral axis
        # --------------------------------------------------------
        grey_out <- t(
            sapply(
                grey_idx,
                function(i)
                    lut[i, i, i, ]
            )
        )
        
        # Draw each segment independently so that
        # its colour corresponds to the output RGB.
        for (i in seq_len(nrow(grey_out) - 1)) {
            col <- rgb(
                pmax(
                    0,
                    pmin(
                        1,
                        grey_out[i, 1]
                    )
                ),
                
                pmax(
                    0,
                    pmin(
                        1,
                        grey_out[i, 2]
                    )
                ),
                
                pmax(
                    0,
                    pmin(
                        1,
                        grey_out[i, 3]
                    )
                )
            )
            
            segments3d(
                x = grey_out[i:(i + 1), 1],
                y = grey_out[i:(i + 1), 2],
                z = grey_out[i:(i + 1), 3],
                col = col,
                lwd = 10
            )
        }
        
        # Deactivated because they were a bit messy. We already have the segments
        # --------------------------------------------------------
        # Points along transformed neutral axis
        # --------------------------------------------------------
        # spheres3d(
        #     grey_out[, 1],
        #     grey_out[, 2],
        #     grey_out[, 3],
        #     radius = 0.012,
        #     color = rgb(
        #         pmax(
        #             0,
        #             pmin(
        #                 1,
        #                 grey_out[, 1]
        #             )
        #         ),
        #         
        #         pmax(
        #             0,
        #             pmin(
        #                 1,
        #                 grey_out[, 2]
        #             )
        #         ),
        #         
        #         pmax(
        #             0,
        #             pmin(
        #                 1,
        #                 grey_out[, 3]
        #             )
        #         )
        #     )
        # )
    }
    
    # ==========================================================
    # INPUT -> OUTPUT VECTORS
    # ==========================================================
    if (show_vectors) {
        vi <- seq(
            1,
            N,
            by = vector_step
        )
        
        for (r in vi) {
            for (g in vi) {
                for (b in vi) {
                    # Input point
                    p0 <- c(
                        u[r],
                        u[g],
                        u[b]
                    )
                    
                    # Output point
                    p1 <- lut[r, g, b, ]
                    
                    # Vector
                    segments3d(
                        x = c(
                            p0[1],
                            p1[1]
                        ),
                        
                        y = c(
                            p0[2],
                            p1[2]
                        ),
                        
                        z = c(
                            p0[3],
                            p1[3]
                        ),
                        
                        col = adjustcolor("grey70", alpha.f = 0.25),
                        lwd = 1
                    )
                }
            }
        }
    }
    
    # ==========================================================
    # LUT OUTPUT VERTICES
    # ==========================================================
    if (show_vertices) {
        combinations <- rbind(
            c(1, 1, 1),
            c(N, 1, 1),
            c(N, N, 1),
            c(1, N, 1),
            
            c(1, 1, N),
            c(N, 1, N),
            c(N, N, N),
            c(1, N, N)
        )
        
        output_vertices <- t(
            sapply(
                seq_len(8),
                function(i)
                    lut[
                        combinations[i, 1],
                        combinations[i, 2],
                        combinations[i, 3],
                    ]
            )
        )
        
        spheres3d(
            output_vertices[, 1],
            output_vertices[, 2],
            output_vertices[, 3],
            radius = 0.02, #0.018,
            color = rgb(
                pmax(
                    0,
                    pmin(
                        1,
                        output_vertices[, 1]
                    )
                ),
                
                pmax(
                    0,
                    pmin(
                        1,
                        output_vertices[, 2]
                    )
                ),
                
                pmax(
                    0,
                    pmin(
                        1,
                        output_vertices[, 3]
                    )
                )
            )
        )
    }
    
    # ==========================================================
    # AXES
    # ==========================================================
    axes3d(
        edges = c(
            "x--",
            "y--",
            "z--"
        ),
        labels = TRUE,
        tick = TRUE
    )
    
    title3d(
        xlab = "R",
        ylab = "G",
        zlab = "B"
    )
    
    # Equal scale on all three axes
    aspect3d(
        1,
        1,
        1
    )
    
    # Initial viewing angle
    view3d(
        theta = 35,
        phi = 25,
        zoom = 0.85
    )
    
    invisible(lut)
}



##################################
# EJEMPLO 1: PROCESADO COLOR RADICAL

# LUT 3D 36x36x36 -> 216x216 HaldCLUT
build_lut("input.tif", "output.tif", "lut", N = 36, force_0=TRUE, force_1=TRUE)

# Apply a .cube LUT
apply_lut(
    input_image_path  = "input_lite.tif",
    lut_path          = "lut.cube",
    output_image_path = "output_lite.tif"
)

# Visualization

# Básica
plot_lut("lut.cube", grid = 3)

# Más densa
plot_lut("lut.cube", grid = 10, line_width = 2)

# Para visualizar además los vectores de transformación (cómo se desplaza cada punto)
plot_lut("lut.cube", grid = 7, line_width = 1, show_vectors = TRUE, vector_step = 4)


##################################
# EJEMPLO 2: PROCESADO BN

# LUT 3D 36x36x36 -> 216x216 HaldCLUT
build_lut("input.tif", "outputbn.tif", "lut_bn", N = 36, force_0=TRUE, force_1=TRUE)

# Apply a .cube LUT
apply_lut(
    input_image_path  = "input_lite.tif",
    lut_path          = "lut_bn.cube",
    output_image_path = "output_lite_BN.tif"
)

# Visualization
plot_lut(
    "lut_bn.cube",
    grid = 10,
    line_width = 1,
    show_vectors = TRUE,
    vector_step = 4
)

