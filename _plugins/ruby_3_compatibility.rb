# Monkey-patch for Ruby 3.2+ compatibility with Jekyll 3.9.x
# Ruby 3.2 removed the `tainted?` method from String, which Jekyll 3.9 relies on.

if RUBY_VERSION >= '3.2'
  class Object
    def tainted?
      false
    end

    def untaint
      self
    end

    def taint
      self
    end
  end

  class String
    def tainted?
      false
    end
  end
end
