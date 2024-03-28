library(shiny)

# Define UI for application that draws a histogram
fluidPage(
  includeCSS("CSS/styles.css"),
  navbarPage( title = "Weterynarz",
    tabPanel(img(src = "Logo.jpg", class = "navbar-img")),
    tabPanel("Panel 1", "Zawartość panelu 1"),
    tabPanel("Panel 2", "Zawartość panelu 2")  
  )
  ,
  # Sidebar with a slider input for number of bins
  sidebarLayout(
    sidebarPanel(
      sliderInput("bins",
                  "Number of bins:",
                  min = 1,
                  max = 50,
                  value = 30)
    ),
    # Show a plot of the generated distribution
    mainPanel(
      plotOutput("distPlot")
    )
  )
)
